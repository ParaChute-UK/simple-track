import warnings
from os import makedirs
from os.path import isdir

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from .object_tracking import interpolate_speeds, ffttrack


class Storm:
    def __init__(self, storm_idx, storm_label_idx, frame, num_dt, under_t, storm_history=False):
        storm_mask = np.where(frame.storm_labels == storm_label_idx)
        self.storm_label_idx = int(storm_label_idx)
        self.area = int(np.size(storm_mask, 1))
        if frame is None:
            return

        if under_t:
            self.extreme = np.min(frame.field[storm_mask])
        else:
            self.extreme = np.max(frame.field[storm_mask])
        self.meanfield = np.mean(frame.field[storm_mask])
        self.centroidx = np.mean(frame.x[storm_mask])
        self.centroidy = np.mean(frame.y[storm_mask])
        self.boxleft = np.min(frame.x[storm_mask])
        self.boxup = np.max(frame.y[storm_mask])
        self.boxwidth = np.max(frame.x[storm_mask]) - np.min(frame.x[storm_mask])
        self.boxheight = np.max(frame.y[storm_mask]) - np.min(frame.y[storm_mask])
        self.life = 1
        if storm_history:
            self.storm_idx = int(storm_label_idx)
            self.dx = np.mean(frame.u[storm_mask]) / num_dt
            self.dy = np.mean(frame.v[storm_mask]) / num_dt
        else:  # First image to be considered, so no dx or dy or previous label
            self.storm_idx = storm_idx
            self.dx = 0
            self.dy = 0
        self.parent = []
        self.child = None
        self.accreted = []

        self.wasdist = None

    @classmethod
    def from_string(cls, storm_label_idx, string):
        storm = cls(storm_label_idx, 0, None, 0, False)
        storm.storm_label_idx = int(storm_label_idx)
        storm.area = int([d for d in string.split() if d.startswith('area=')][0].replace('area=', ''))
        storm.extreme = float([d for d in string.split() if d.startswith('extreme=')][0].replace('extreme=', ''))
        storm.meanfield = float([d for d in string.split() if d.startswith('meanv=')][0].replace('meanv=', ''))
        storm.centroidx = float(
            [d for d in string.split() if d.startswith('centroid=')][0].replace('centroid=', '').split(',')[0]
        )
        storm.centroidy = float(
            [d for d in string.split() if d.startswith('centroid=')][0].replace('centroid=', '').split(',')[1]
        )
        storm.life = int([d for d in string.split() if d.startswith('life=')][0].replace('life=', ''))
        storm.storm_idx = int(string.split()[1])
        storm.dx = float([d for d in string.split() if d.startswith('dx=')][0].replace('dx=', ''))
        storm.dy = float([d for d in string.split() if d.startswith('dy=')][0].replace('dy=', ''))
        storm.parent = [
            int(p) for p in [d for d in string.split() if d.startswith('parent=')][0].replace('parent=', '').split(',')
        ]
        storm.child = [
            int(p) for p in [d for d in string.split() if d.startswith('child=')][0].replace('child=', '').split(',')
        ]
        storm.accreted = [
            int(p)
            for p in [d for d in string.split() if d.startswith('accreted=')][0].replace('accreted=', '').split(',')
        ]
        box = [d for d in string.split() if d.startswith('box=')][0].replace('box=', '').split(',')
        storm.boxleft = float(box[0])
        storm.boxup = float(box[1])
        storm.boxheight = float(box[2])
        storm.boxwidth = float(box[3])

    def inherit_properties(
        self, frame, storm_label_idx, old_storm_data, kindex, qhist, lapthresh, single_overlap=False
    ):
        self.storm_idx = old_storm_data[kindex].storm_idx
        self.life = old_storm_data[kindex].life + 1
        # TODO: ??
        self.wasdist = np.size(
            np.where((frame.propagated_storm_labels == kindex + 1) & (frame.storm_labels == storm_label_idx)), 1
        )
        # Handle multiple clouds overlap
        if not single_overlap:
            alllaps = np.where(qhist[1:] >= lapthresh)
            for kkind in range(np.size(alllaps, 1)):
                allindex = np.squeeze(alllaps[0][kkind])
                if allindex == kindex:
                    continue
                self.accreted.append(old_storm_data[allindex].storm_idx)


class Frame:
    def __init__(self, time, field, x, y):
        self.time = time
        self.field = field
        self.x = x
        self.y = y

        self.storm_labels = None
        self.storms_mask = None
        self.numstorms = 0
        self.storm_data = []

        self.u = None
        self.v = None
        self.advected_storms = None
        self.propagated_storm_labels = None


# TRACKING ALGORITHM
# 1. Correlate previous and current time step to find (dx,dy) displacements.
# 2. Propagate features from previous time step to current time step using (dx,dy) displacements.
# 3. Iterate through objects to check for overlap and inherit object properties.
# 4. Iterate through objects to check for splitting and merging events.


class StormTracker:
    def __init__(
        self,
        loader,
        outdir,
        dt=5.0,
        dt_tolerance=15.0,
        # Maximum separation in time allowed between consecutive images
        under_t=False,
        ## True = labelling areas *under* the threshold (e.g. brightness temperature), False = labelling areas *above* threshold (e.g. rainfall)
        threshold=3.0,
        ## Threshold used to identify objects (with value of variable greater than this threshold)
        minpixel=4.0,  ## Minimum object size in pixels
        squarelength=200.0,
        ## Size in pixels of individual squares to run fft for (dx,dy) displacement. Must divide (x,y) lengths of the array!
        rafraction=0.01,
        ## Minimum fractional cover of objects required for fft to obtain (dx,dy) displacement
        dd_tolerance=3.0,
        # Maximum difference in displacement values between adjacent squares (to remove spurious values) - scaled by num_dt if necessary
        halopixel=5.0,
        ## Radius of halo in pixels for orphan storms - big halo assumes storms may spawn "children" at a distance multiple pixels away
        flagwrite=True,  ## For writing storm history data in a text file
        struct2d=np.ones((3, 3)),
        ## np.ones((3,3)) is 8-point connectivity for labelling storms. Can be changed to user preference.
        lapthresh=0.6,  ## Minimum fraction of overlap (0.6 in TITAN)
    ):
        self.loader = loader
        self.outdir = outdir

        self.dt = dt
        self.dt_tolerance = dt_tolerance
        self.under_t = under_t
        self.threshold = threshold
        self.minpixel = minpixel
        self.squarelength = squarelength
        self.rafraction = rafraction
        self.dd_tolerance = dd_tolerance
        self.halopixel = halopixel
        self.flagwrite = flagwrite
        self.struct2d = struct2d
        self.lapthresh = lapthresh

        self.tukey_window = 1
        self.squarehalf = int(squarelength / 2)
        self.fftpixels = squarelength**2 / int(1.0 / rafraction)
        self.halosq = halopixel**2
        self.x, self.y = np.meshgrid(range(-400, 400), range(-300, 300))
        xall = np.size(self.x, 0)  # Only used to check grid dimensions
        yall = np.size(self.x, 1)  # Only used to check grid dimensions
        if np.fmod(xall, squarelength) != 0 or np.fmod(yall, squarelength) != 0:
            raise ValueError('Your grid does not match a multiple of squares as defined by squarelength')

        self.new_storm_idx = 1
        self.num_dt = None

    def track_storms(self):
        old_frame = None
        times = []
        frames = []
        for nt, (field, file_id, now_time) in enumerate(self.loader.load_next()):
            if nt == 3:
                break
            print(now_time)
            times.append(now_time)
            frame = Frame(now_time, field, self.x, self.y)
            # 0. label storms based on threshold.
            self.label_storms(frame)

            if old_frame is not None:
                # check time difference between consecutive images
                dtnow = (now_time - old_frame.time).total_seconds() / 60  # timediff in minutes.
                self.num_dt = dtnow / self.dt

                if dtnow > self.dt_tolerance:
                    print('Data are too far apart in time --- Re-initialise objects')
                    old_frame = None
                    continue

            if old_frame is None:
                for i in range(frame.numstorms):
                    storm_label_idx = i + 1  # First storm is labelled 1, but python indeces start at 0.
                    frame.storm_data.append(
                        Storm(
                            self.new_storm_idx, storm_label_idx, frame, self.num_dt, self.under_t, storm_history=False
                        )
                    )
                    self.new_storm_idx += 1
            elif old_frame.numstorms and frame.numstorms:
                # 1. Calculate correlation velocites by comparing old_frame and frame.
                self.calc_corr_velocities(
                    old_frame, frame, self.fftpixels, self.num_dt, self.squarehalf, self.tukey_window
                )

                # 2. Assign displacement to each of the old storms.
                self.assign_displacements(old_frame, frame)

                # 3. Find overlaps between the advected storms and current storms.
                storm_idxs = self.find_overlaps(old_frame, frame)

                # 4. Check for splitting and merging events.
                self.check_multiple_merges(frame, storm_idxs)

            self.write_output_text(times[0], frame, file_id)
            frames.append(frame)
            old_frame = frame

        print(frames)
        self.write_output(times, frames)

    def write_output(self, times, frames):
        storms = [storm for frame in frames for storm in frame.storm_data]
        storm_idxs = [s.storm_idx for s in storms]
        ds = xr.Dataset(
            data_vars={
                'storm_labels': (['time', 'northings', 'eastings'], np.array([f.storm_labels for f in frames])),
            },
            coords={
                'time': times,
                'northings': self.loader.curr_da.northings,
                'eastings': self.loader.curr_da.eastings,
                'storm_idx': storm_idxs,
            },
        )
        print(ds)
        cols = [
            'storm_idx',
            'storm_label_idx',
            'life',
            'area',
            'extreme',
            'meanfield',
            'centroidx',
            'centroidy',
            'boxleft',
            'boxup',
            'boxwidth',
            'boxheight',
            'dx',
            'dy',
        ]
        storm_times = [time for frame in frames for time in [frame.time] * frame.numstorms]
        df = pd.DataFrame(data={col: [getattr(s, col) for s in storms] for col in cols})
        df.insert(1, 'time', np.array(storm_times))
        print(df)
        print(df[df.storm_idx == 88])

        # print("images_dir + file_id +'.txt'=", images_dir + file_id +'.txt')
        print(self.outdir + 'storm_labels.nc')
        ds.to_netcdf(self.outdir + 'storm_labels.nc')
        print(self.outdir + 'storm_data.hdf')
        df.to_hdf(self.outdir + 'storm_data.hdf', key='storm_data')

        # self.parent = []
        # self.child = None
        # self.accreted = []

        # self.wasdist = None

    def label_storms(self, frame):
        binfield = np.zeros_like(frame.field)
        if self.under_t:
            binfield[np.where(frame.field < self.threshold)] = 1
        else:
            binfield[np.where(frame.field > self.threshold)] = 1
        # Perform full labelling
        id_regions, num_ids = ndimage.label(binfield, structure=self.struct2d)
        # Apply area threshold.
        id_sizes = np.array(ndimage.sum(binfield, id_regions, range(num_ids + 1)))
        area_mask = id_sizes < self.minpixel
        binfield[area_mask[id_regions]] = 0

        frame.storm_labels, num_ids = ndimage.label(binfield, structure=self.struct2d)
        frame.numstorms = num_ids  # TODO: +1?
        frame.storms_mask = frame.storm_labels > 0
        print('num_ids = ', num_ids)

    def calc_corr_velocities(
        self,
        old_frame,
        frame,
        fftpixels,
        num_dt,
        squarehalf,
        tukey_window,
    ):
        # old_storm_data & storm_data are not empty, so use fft to get velocities
        # and update uvlabel in old_storm_data accordingly
        # Estimate velocities using squares within domain
        xint, yint = np.meshgrid(
            range(frame.x[0, 0] + squarehalf, frame.x[0, -1], squarehalf),
            range(frame.y[0, 0] + squarehalf, frame.y[-1, 0], squarehalf),
        )
        buu = np.full(xint.shape, np.NaN)
        bvv = np.full(xint.shape, np.NaN)
        bww = np.full(xint.shape, np.NaN)
        for corx in range(0, int(np.size(xint, 0))):
            for cory in range(0, int(np.size(xint, 1))):
                oldsquare = old_frame.storms_mask[
                    squarehalf * corx : squarehalf * corx + 2 * squarehalf,
                    squarehalf * cory : squarehalf * cory + 2 * squarehalf,
                ]
                newsquare = frame.storms_mask[
                    squarehalf * corx : squarehalf * corx + 2 * squarehalf,
                    squarehalf * cory : squarehalf * cory + 2 * squarehalf,
                ]
                if (
                    np.sum(oldsquare) < fftpixels or np.sum(newsquare) < fftpixels
                ):  # if there are too few storms, don't try to derive motion vectors.
                    buu[corx, cory] = np.NaN
                    bvv[corx, cory] = np.NaN
                    bww[corx, cory] = np.NaN
                else:
                    dx, dy, amplitude, corrval = ffttrack(oldsquare, newsquare, tukey_window)
                    buu[corx, cory] = dx
                    bvv[corx, cory] = dy  ## indices are upside down so need minus to get real-world dy-velocity
                    bww[corx, cory] = amplitude
        # check neighbouring values for smoothness
        # Ignore warnings about mean over empty array in this section
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for corx in range(0, int(np.size(xint, 0))):
                for cory in range(0, int(np.size(xint, 1))):
                    if np.isnan(buu[corx, cory]) and np.isnan(bvv[corx, cory]):
                        continue
                    if corx == 0:
                        if cory == 0:
                            bu_nb = np.nanmean([buu[0, 1], buu[1, 0], buu[1, 1]])
                            bv_nb = np.nanmean([bvv[0, 1], bvv[1, 0], bvv[1, 1]])
                        elif cory == int(np.size(xint, 1)) - 1:
                            bu_nb = np.nanmean([buu[0, cory - 1], buu[1, cory], buu[1, cory - 1]])
                            bv_nb = np.nanmean([bvv[0, cory - 1], bvv[1, cory], bvv[1, cory - 1]])
                        else:
                            bu_nb = np.nanmean(
                                [buu[0, cory + 1], buu[0, cory - 1], buu[1, cory - 1], buu[1, cory], buu[1, cory + 1]]
                            )
                            bv_nb = np.nanmean(
                                [bvv[0, cory + 1], bvv[0, cory - 1], bvv[1, cory - 1], bvv[1, cory], bvv[1, cory + 1]]
                            )
                    elif corx == int(np.size(xint, 0)) - 1:
                        if cory == 0:
                            bu_nb = np.nanmean([buu[corx, 1], buu[corx - 1, 0], buu[corx - 1, 1]])
                            bv_nb = np.nanmean([bvv[corx, 1], bvv[corx - 1, 0], bvv[corx - 1, 1]])
                        elif cory == int(np.size(xint, 1)) - 1:
                            bu_nb = np.nanmean([buu[corx, cory - 1], buu[corx - 1, cory], buu[corx - 1, cory - 1]])
                            bv_nb = np.nanmean([bvv[corx, cory - 1], bvv[corx - 1, cory], bvv[corx - 1, cory - 1]])
                        else:
                            bu_nb = np.nanmean(
                                [
                                    buu[corx, cory + 1],
                                    buu[corx, cory - 1],
                                    buu[corx - 1, cory - 1],
                                    buu[corx - 1, cory],
                                    buu[corx - 1, cory + 1],
                                ]
                            )
                            bv_nb = np.nanmean(
                                [
                                    bvv[corx, cory + 1],
                                    bvv[corx, cory - 1],
                                    bvv[corx - 1, cory - 1],
                                    bvv[corx - 1, cory],
                                    bvv[corx - 1, cory + 1],
                                ]
                            )
                    elif cory == int(np.size(xint, 1)) - 1:
                        bu_nb = np.nanmean(
                            [
                                buu[corx, cory - 1],
                                buu[corx - 1, cory],
                                buu[corx - 1, cory - 1],
                                buu[corx + 1, cory - 1],
                                buu[corx + 1, cory],
                            ]
                        )
                        bv_nb = np.nanmean(
                            [
                                bvv[corx, cory - 1],
                                bvv[corx - 1, cory],
                                bvv[corx - 1, cory - 1],
                                bvv[corx + 1, cory - 1],
                                bvv[corx + 1, cory],
                            ]
                        )
                    else:
                        bu_nb = np.nanmean(
                            [
                                buu[corx, cory + 1],
                                buu[corx, cory - 1],
                                buu[corx - 1, cory - 1],
                                buu[corx - 1, cory],
                                buu[corx - 1, cory + 1],
                                buu[corx + 1, cory - 1],
                                buu[corx + 1, cory],
                                buu[corx + 1, cory + 1],
                            ]
                        )
                        bv_nb = np.nanmean(
                            [
                                bvv[corx, cory + 1],
                                bvv[corx, cory - 1],
                                bvv[corx - 1, cory - 1],
                                bvv[corx - 1, cory],
                                bvv[corx - 1, cory + 1],
                                bvv[corx + 1, cory - 1],
                                bvv[corx + 1, cory],
                                bvv[corx + 1, cory + 1],
                            ]
                        )
                    if np.abs(buu[corx, cory] - bu_nb) > self.dd_tolerance * num_dt:
                        buu[corx, cory] = np.nan
                    if np.abs(bvv[corx, cory] - bv_nb) > self.dd_tolerance * num_dt:
                        bvv[corx, cory] = np.nan
        # ACTUAL DISPLACEMENT
        # Interpolate these displacements onto the full grid
        frame.u = interpolate_speeds(xint, yint, frame.x, frame.y, buu, old_frame.storm_labels)
        frame.v = interpolate_speeds(xint, yint, frame.x, frame.y, bvv, old_frame.storm_labels)

    def assign_displacements(self, old_frame, frame):
        frame.propagated_storm_labels = np.zeros(old_frame.storm_labels.shape)
        for i in range(len(old_frame.storm_data)):
            storm_label_idx = old_frame.storm_data[i].storm_label_idx
            labelind = np.where(old_frame.storm_labels == storm_label_idx)
            dx = np.mean(frame.u[labelind])
            dy = np.mean(frame.v[labelind])
            if dx == 0.0 and dy == 0.0:
                frame.propagated_storm_labels[labelind] = storm_label_idx
            else:
                for ii in range(np.size(labelind, 1)):
                    newyind = labelind[1][ii] + int(np.around(dx))
                    newxind = labelind[0][ii] + int(np.around(dy))
                    if (
                        newxind > np.size(frame.propagated_storm_labels, 0) - 1
                        or newyind > np.size(frame.propagated_storm_labels, 1) - 1
                        or newxind < 0
                        or newyind < 0
                    ):
                        continue
                    elif frame.propagated_storm_labels[newxind, newyind] > 0:
                        nq = int(frame.propagated_storm_labels[newxind, newyind] - 1)
                        olddist = (frame.x[newxind, newyind] - old_frame.storm_data[nq].centroidx) ** 2 + (
                            frame.y[newxind, newyind] - old_frame.storm_data[nq].centroidy
                        ) ** 2
                        newdist = (frame.x[newxind, newyind] - old_frame.storm_data[i].centroidx) ** 2 + (
                            frame.y[newxind, newyind] - old_frame.storm_data[i].centroidy
                        ) ** 2
                        if newdist < olddist:
                            frame.propagated_storm_labels[newxind, newyind] = storm_label_idx
                    else:
                        frame.propagated_storm_labels[newxind, newyind] = storm_label_idx
        frame.advected_storms = np.zeros([len(old_frame.storm_data), 3])
        for i in range(len(old_frame.storm_data)):
            storm_label_idx = old_frame.storm_data[i].storm_label_idx
            centrind = np.where(frame.propagated_storm_labels == storm_label_idx)
            if np.size(centrind, 1) == 0:
                continue
            else:
                frame.advected_storms[i][0] = np.mean(frame.x[centrind])
                frame.advected_storms[i][1] = np.mean(frame.y[centrind])
                frame.advected_storms[i][2] = int(np.size(centrind, 1))

    def find_overlaps(self, old_frame, frame):
        # Loop through storm_data and check for overlap with
        # advected old_storm_data storms
        qbins = range(int(np.max(old_frame.storm_labels)) + 2)
        qarea = np.ones([int(np.max(old_frame.storm_labels)) + 1])
        qlife = np.ones([int(np.max(old_frame.storm_labels)) + 1])
        for qq in range(len(old_frame.storm_data)):
            if frame.advected_storms[qq, 2] > 0:
                qarea[qq + 1] = frame.advected_storms[qq, 2]
            qlife[qq + 1] = old_frame.storm_data[qq].life
        for i in range(frame.numstorms):
            storm_label_idx = i + 1  # first storm is labelled 1, but python indeces start at 0.
            frame.storm_data.append(
                Storm(self.new_storm_idx, storm_label_idx, frame, self.num_dt, self.under_t, storm_history=True)
            )

            # check overlap with qhist
            # if no overlap, then
            # generate (halo) km radius around centroid
            # check for overlap within (halo) km of centroid
            qhist = (
                np.histogram(frame.propagated_storm_labels[np.where(frame.storm_labels == storm_label_idx)], qbins)
            )[0][:] / float(frame.storm_data[i].area) + (
                np.histogram(frame.propagated_storm_labels[np.where(frame.storm_labels == storm_label_idx)], qbins)
            )[
                0
            ][
                :
            ] / qarea[
                :
            ]

            if np.max(qhist[1:]) < self.lapthresh:
                newblob = 0 * frame.x
                blobind = np.where(
                    (frame.x - frame.storm_data[i].centroidx) ** 2 + (frame.y - frame.storm_data[i].centroidy) ** 2
                    < self.halosq
                )
                newblob[blobind] = newblob[blobind] + 1
                qhist = (np.histogram(frame.propagated_storm_labels[np.where(newblob == 1)], qbins))[0][:] / float(
                    frame.storm_data[i].area
                ) + (np.histogram(frame.propagated_storm_labels[np.where(newblob == 1)], qbins))[0][:] / qarea[:]
            # if overlap, then
            # - inherit "was"
            # - update "life" and "track" and "wasdist"
            # - inherit "dx" and "dy" (only update if single overlap)
            if np.max(qhist[1:]) >= self.lapthresh:
                numlaps = np.where(qhist[1:] >= self.lapthresh)
                # if more than one good overlap
                # keep properties of storm with largest overlap
                # if more than one largest, keep nearest in centroid
                if np.size(numlaps, 1) > 1:
                    lapdist = np.zeros([np.size(numlaps, 1)])
                    sectlap = np.zeros([np.size(numlaps, 1)])
                    for kkind in range(np.size(numlaps, 1)):
                        qindex = np.squeeze(numlaps[0][kkind])
                        lapdist[kkind] = np.sqrt(
                            (frame.storm_data[i].centroidx - frame.advected_storms[qindex, 0]) ** 2
                            + (frame.storm_data[i].centroidy - frame.advected_storms[qindex, 1]) ** 2
                        )
                        sectlap[kkind] = np.size(
                            np.where(
                                (frame.propagated_storm_labels == qindex + 1) & (frame.storm_labels == storm_label_idx)
                            ),
                            1,
                        )
                    kmax = np.where(sectlap == np.max(sectlap))
                    if np.size(kmax, 1) > 1:
                        kkmax = kmax[0][np.where(lapdist[kmax[0][:]] == np.min(lapdist[kmax[0][:]]))]
                        if np.size(kkmax) > 1:
                            kkmax = kmax[0][kkmax[0]]
                    else:
                        kkmax = kmax[0][0]
                    kindex = np.squeeze(numlaps[0][kkmax])
                    frame.storm_data[i].inherit_properties(
                        frame,
                        storm_label_idx,
                        old_frame.storm_data,
                        kindex,
                        qhist,
                        self.lapthresh,
                        single_overlap=False,
                    )
                else:
                    # single overlap
                    zindex = np.squeeze(numlaps[0][0])
                    frame.storm_data[i].inherit_properties(
                        frame, storm_label_idx, old_frame.storm_data, zindex, qhist, self.lapthresh, single_overlap=True
                    )

            else:
                # if no overlap, then (new storm)
                # - "storm_idx" set to current max label +1
                # - update "life" for a new storm
                frame.storm_data[i].storm_idx = self.new_storm_idx
                frame.storm_data[i].life = 1
                self.new_storm_idx += 1
        storm_idxs = np.array([frame.storm_data[i].storm_idx for i in range(len(frame.storm_data))])

        # ensure that accreted should never be a value similar to existing storm id
        for i in range(len(frame.storm_data)):
            if not frame.storm_data[i].accreted:
                continue
            else:
                acnum = None
                for acnum in range(np.size(frame.storm_data[i].accreted)):
                    acind = np.where((storm_idxs - frame.storm_data[i].accreted[acnum]) == 0)
                    if np.size(acind, 1) > 0:
                        frame.storm_data[i].accreted[acnum] = None  # TODO: ??
                # acnew=np.where(storm_data[i].accreted > misval)
                acnew = [aci for aci in frame.storm_data[i].accreted if aci is not None]
                if np.size(acnew) > 0:
                    for acindex in range(np.size(acnew)):
                        frame.storm_data[i].accreted[acnum] = acnew[acindex]
                else:
                    frame.storm_data[i].accreted = []
        return storm_idxs

    def check_multiple_merges(self, frame, storm_idxs):
        # tracking merging breaking
        # multiple storms at t (storm_data) may have same label "was"
        # find storm with largest overlap at t+1 with advected Q(t)
        # this is the "parent" storm,
        # "parent" vector with indices of new labels for "child" storms
        # storms with same was but futher from centroid are "child", value "parent"
        for i in range(len(frame.storm_data)):
            if frame.storm_data[i].wasdist is None:
                continue
            wasind = np.where(storm_idxs == storm_idxs[i])
            wasseplength = 0
            for kkind in range(np.size(wasind)):
                if frame.storm_data[wasind[0][kkind]].wasdist is None:
                    continue
                else:
                    wasseplength = wasseplength + 1
            wassep = np.zeros(wasseplength)
            if np.size(wasind) > 1:
                kkval = 0
                for kkind in range(np.size(wasind)):
                    if frame.storm_data[wasind[0][kkind]].wasdist is None:
                        continue
                    else:
                        wassep[kkval] = frame.storm_data[wasind[0][kkind]].wasdist
                        kkval = kkval + 1
            else:
                wassep = frame.storm_data[wasind[0][0]].wasdist
            # wassep now contains all non-zero overlap values
            # find the maximum (this will be the parent)
            # all other storms will be the children
            kmax = np.where(wassep == np.max(wassep))
            kkmax = np.min(kmax)
            children = []
            for kkind in range(np.size(wassep)):
                if not kkind == kkmax:
                    frame.storm_data[wasind[0][kkind]].child = frame.storm_data[wasind[0][kkmax]].storm_idx
                    frame.storm_data[wasind[0][kkind]].storm_idx = self.new_storm_idx
                    frame.storm_data[wasind[0][kkind]].life = frame.storm_data[wasind[0][kkmax]].life
                    self.new_storm_idx += 1
                    storm_idxs[wasind[0][kkind]] = frame.storm_data[wasind[0][kkind]].storm_idx
                    children.append(frame.storm_data[wasind[0][kkind]].storm_idx)
                    frame.storm_data[wasind[0][kkind]].wasdist = None
            # update parent storm with children
            if np.size(children) > 0:
                frame.storm_data[wasind[0][kkmax]].parent = children

    def write_output_text(self, init_time, frame, file_id):
        if not (isdir(self.outdir)):
            makedirs(self.outdir)
        label_method = 'Rainfall rate > ' + str(int(self.threshold)) + 'mm/hr'
        domain = 'chil_' if True else 'central_'
        write_file_id = (
            domain
            + 'S'
            + str(int(self.squarelength))
            + '_T'
            + str(int(self.threshold))
            + '_A'
            + str(int(self.minpixel))
            + '_'
            + file_id
        )

        # print("images_dir + file_id +'.txt'=", images_dir + file_id +'.txt')
        fw = open(self.outdir + 'history_' + write_file_id + '.txt', 'w')
        fw.write('missing_value=' + str(-999) + '\r\n')
        fw.write('Start date and time=' + init_time.strftime('%d/%m/%y-%H%M') + '\r\n')
        fw.write('Current date and time=' + frame.time.strftime('%d/%m/%y-%H%M') + '\r\n')
        fw.write('Label method=' + label_method + '\r\n')
        fw.write('Squarelength=' + str(self.squarelength) + '\r\n')
        fw.write('Rafraction=' + str(self.rafraction) + '\r\n')
        fw.write('total number of tracked storms=' + str(self.new_storm_idx - 1) + '\r\n')
        for ns in range(len(frame.storm_data)):
            fw.write('storm ' + str(frame.storm_data[ns].storm_idx))
            fw.write(' area=' + str(frame.storm_data[ns].area))
            fw.write(
                ' centroid='
                + str(round(frame.storm_data[ns].centroidx, 2))
                + ','
                + str(round(frame.storm_data[ns].centroidy, 2))
            )
            fw.write(
                ' box='
                + str(frame.storm_data[ns].boxleft)
                + ','
                + str(frame.storm_data[ns].boxup)
                + ','
                + str(frame.storm_data[ns].boxwidth)
                + ','
                + str(frame.storm_data[ns].boxheight)
            )
            fw.write(' life=' + str(frame.storm_data[ns].life))
            fw.write(' dx=' + str(round(frame.storm_data[ns].dx, 2)) + ' dy=' + str(round(frame.storm_data[ns].dy, 2)))

            fw.write(' meanv=' + str(round(frame.storm_data[ns].meanfield, 2)))
            fw.write(' extreme=' + str(round(frame.storm_data[ns].extreme, 2)) + ' accreted=')
            if np.size(frame.storm_data[ns].accreted) > 0:
                for acind in range(np.size(frame.storm_data[ns].accreted) - 1):
                    if frame.storm_data[ns].accreted[acind] is None:
                        fw.write(str(-999) + ',')
                    else:
                        fw.write(str(frame.storm_data[ns].accreted[acind]) + ',')
                fw.write(str(frame.storm_data[ns].accreted[np.size(frame.storm_data[ns].accreted) - 1]) + ' parent=')
            else:
                fw.write(str(-999) + ' parent=')
            child = frame.storm_data[ns].child
            child_str = str(-999) if child is None else str(child)
            fw.write(child_str + ' child=')
            # fw.write(str(frame.storm_data[ns].child) + ' child=')
            if np.size(frame.storm_data[ns].parent) > 0:
                for acind in range(np.size(frame.storm_data[ns].parent) - 1):
                    fw.write(str(frame.storm_data[ns].parent[acind]) + ',')
                fw.write(str(frame.storm_data[ns].parent[np.size(frame.storm_data[ns].parent) - 1]) + '\r\n')
            else:
                fw.write(str(-999) + '\r\n')
        fw.close()
