#!/usr/local/sci/bin/python2.7

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate
import scipy.ndimage as ndimage
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from skimage.registration import phase_cross_correlation
from scipy.signal.windows import hann

###################################################################
# Initiate StormS class with object properties. Can be adjusted to store additional object properties.
# Future release could relate StormS and write_storms functions for easier management and user adaptation.
###################################################################


class StormS:
    def __init__(
        self,
        jj,  # ??
        StormLabels,
        var,
        xmat,
        ymat,
        newwas,  # Still don't know what this is!!
        newumat,
        newvmat,
        num_dt,
        misval,
        doradar,
        under_threshold,
        extra_thresh=[],
        storm_history=False,
        string=None,  # Input from previous txt file
        rarray=[],
        azarray=[],
    ):
        if string is None:  # initialiase to default values
            C = np.where(StormLabels == jj)
            self.storm = int(jj)
            self.area = int(np.size(C, 1))
            if under_threshold:
                self.extreme = np.min(var[C])
            else:
                self.extreme = np.max(var[C])
            self.meanvar = np.mean(var[C])
            if len(extra_thresh) > 0:
                self.extra_area = [(var[C] < a).sum() for a in extra_thresh]
            self.centroidx = np.mean(xmat[C])
            self.centroidy = np.mean(ymat[C])
            self.boxleft = np.min(xmat[C])
            self.boxup = np.max(ymat[C])
            self.boxwidth = np.max(xmat[C]) - np.min(xmat[C])
            self.boxheight = np.max(ymat[C]) - np.min(ymat[C])
            self.life = 1
            if storm_history:
                self.was = int(jj)
                self.dx = np.mean(newumat[C]) / num_dt
                self.dy = np.mean(newvmat[C]) / num_dt
            else:  # First image to be considered, so no dx or dy or previous label
                self.was = newwas  # Still don't know what this is!!
                self.dx = 0.0
                self.dy = 0
            self.parent = [misval]
            self.child = misval
            self.wasdist = misval
            self.accreted = [misval]
            if doradar:
                self.rangel = np.min(rarray[C])
                self.rangeu = np.max(rarray[C])
                if np.min(rarray[C]) == 0:
                    self.azimuthl = 0.0
                    self.azimuthu = 360.0
                elif (np.min(azarray[C]) == 0) & (np.max(azarray[C]) > 180):
                    azxy = np.where(
                        (xmat == np.round(self.centroidx))
                        & (ymat == np.round(self.centroidy))
                    )
                    azoffset = np.fmod(np.round(azarray[azxy]) + 180.0, 360.0)
                    aznotind = np.where((StormLabels == jj) & (azarray < azoffset))
                    if np.size(aznotind) == 0:
                        self.azimuthl = 0.0
                        self.azimuthu = 360.0
                    elif np.max(azarray[aznotind]) > azoffset - 1.0:
                        self.azimuthl = 0.0
                        self.azimuthu = 360.0
                    else:
                        azleftind = np.where(
                            (StormLabels == jj) & (azarray >= azoffset)
                        )
                        if np.size(azleftind) == 0:
                            self.azimuthl = np.min(azarray[aznotind])
                            self.azimuthu = np.max(azarray[aznotind])
                        else:
                            self.azimuthl = np.min(azarray[azleftind])
                            self.azimuthu = np.max(azarray[aznotind])
                else:
                    self.azimuthl = np.min(azarray[C])
                    self.azimuthu = np.max(azarray[C])
        else:  # Use string which is line in save file to initialise
            self.storm = int(jj)
            self.area = int(
                [d for d in string.split() if d.startswith("area=")][0].replace(
                    "area=", ""
                )
            )
            self.extreme = float(
                [d for d in string.split() if d.startswith("extreme=")][0].replace(
                    "extreme=", ""
                )
            )
            self.meanvar = float(
                [d for d in string.split() if d.startswith("meanv=")][0].replace(
                    "meanv=", ""
                )
            )
            if len(extra_thresh) > 0:
                self.extra_area = [
                    int(
                        [d for d in string.split() if d.startswith("area<" + str(e))][
                            0
                        ].split("=")[-1]
                    )
                    for e in extra_thresh
                ]
            self.centroidx = float(
                [d for d in string.split() if d.startswith("centroid=")][0]
                .replace("centroid=", "")
                .split(",")[0]
            )
            self.centroidy = float(
                [d for d in string.split() if d.startswith("centroid=")][0]
                .replace("centroid=", "")
                .split(",")[1]
            )
            self.life = int(
                [d for d in string.split() if d.startswith("life=")][0].replace(
                    "life=", ""
                )
            )
            self.was = int(string.split()[1])
            self.dx = float(
                [d for d in string.split() if d.startswith("dx=")][0].replace("dx=", "")
            )
            self.dy = float(
                [d for d in string.split() if d.startswith("dy=")][0].replace("dy=", "")
            )
            self.parent = [
                int(p)
                for p in [d for d in string.split() if d.startswith("parent=")][0]
                .replace("parent=", "")
                .split(",")
            ]
            self.child = [
                int(p)
                for p in [d for d in string.split() if d.startswith("child=")][0]
                .replace("child=", "")
                .split(",")
            ]
            self.accreted = [
                int(p)
                for p in [d for d in string.split() if d.startswith("accreted=")][0]
                .replace("accreted=", "")
                .split(",")
            ]
            box = (
                [d for d in string.split() if d.startswith("box=")][0]
                .replace("box=", "")
                .split(",")
            )
            self.boxleft = float(box[0])
            self.boxup = float(box[1])
            self.boxheight = float(box[2])
            self.boxwidth = float(box[3])
            if doradar:
                self.range = float(
                    [d for d in string.split() if d.startswith("range=")][0].replace(
                        "range=", ""
                    )
                )
                self.azimuthl = float(
                    [d for d in string.split() if d.startswith("azimuth=")][0].replace(
                        "azimuth=", ""
                    )[0]
                )
                self.azimuthu = [d for d in string.split() if d.startswith("azimuth=")][
                    0
                ].replace("azimuth=", "")[1]

    def inherit_properties(
        self,
        jj,
        OldStormData,
        kindex,
        QuvL,
        StormLabels,
        qhist,
        lapthresh,
        misval,
        single_overlap=False,
    ):
        self.was = OldStormData[kindex].was
        self.life = OldStormData[kindex].life + 1
        # Find the number of points in common between advected and new fields, matched by id...?
        # I.e., the overlap between advected previous feautre and current feature
        self.wasdist = np.size(np.where((QuvL == kindex + 1) & (StormLabels == jj)), 1)
        qind = kindex + 1
        # Code below is only required when multiple clouds overlap
        if not (single_overlap):
            alllaps = np.where(qhist[1:] >= lapthresh)
            for kkind in range(np.size(alllaps, 1)):
                allindex = np.squeeze(alllaps[0][kkind])
                # Don't do anything if the overlaps are just the matched id
                if allindex == kindex:
                    continue
                # Replace misval if it is there
                if self.accreted[-1] == misval:
                    self.accreted[-1] = OldStormData[allindex].was
                # Or, append to existing list. Does this get accessed??
                else:
                    self.accreted.append(OldStormData[allindex].was)


def initialise_first_field(
    field,
    newwas,
    StormLabels,
    xmat,
    ymat,
    squarehalf,
    num_dt,
    misval,
    doradar,
    under_threshold,
    rarray=[],
    azarray=[],
):
    ###############################################################
    # PARAMETERS FOR FUTURE FUNCTIONALITY
    ###################################################################
    extra_thresh = []

    ###############################################################
    # START TRACKING!!
    ###################################################################

    newumat = 0
    newvmat = 0
    wasarray = 0 * StormLabels  # set up array of zeros.
    lifearray = 0 * StormLabels
    numstorms = StormLabels.max()
    print("numstorms = ", numstorms)
    StormData = []

    # Check xmat and ymat have same shape
    np.testing.assert_equal(
        xmat.shape, ymat.shape, "xmat and ymat must have same shape"
    )

    # Check squarehalf input for validity. If it is too large compared to domain size, raise exception
    min_domain_size = np.min((xmat.shape[0], xmat.shape[1]))
    if squarehalf * 2 >= min_domain_size:
        raise ValueError(
            "squarehalf input is too large for domain size. Reduce squarehalf."
        )

    for ns in range(numstorms):
        jj = ns + 1  # First storm is labelled 1, but python indeces start at 0.
        C = np.where(StormLabels == jj)
        # TODO: Is this appending lists to the StormData list?? Unusual!!
        StormData += [
            StormS(
                jj,
                StormLabels,
                field,
                xmat,
                ymat,
                newwas,
                0,
                0,
                num_dt,
                misval,
                doradar,
                under_threshold,
                extra_thresh=extra_thresh,
                storm_history=False,
                string=None,
                rarray=rarray,
                azarray=azarray,
            )
        ]
        wasarray[C] = newwas
        newwas = newwas + 1
        lifearray[C] = 1

    return StormData, newwas, StormLabels, newumat, newvmat, wasarray, lifearray


###################################################################
# TRACKING ALGORITHM
# 1. Correlate previous and current time step to find (dx,dy) displacements.
# 2. Propagate features from previous time step to current time step using (dx,dy) displacements.
# 3. Iterate through objects to check for overlap and inherit object properties.
# 4. Iterate through objects to check for splitting and merging events.
###################################################################


def track_storms(
    OldStormData,
    field,
    newwas,
    StormLabels,
    OldStormLabels,
    xmat,
    ymat,
    fftpixels,
    dd_tolerance,
    halosq,
    squarehalf,
    prev_features,
    current_features,
    num_dt,
    lapthresh,
    misval,
    doradar,
    under_threshold,
    IMAGES_DIR,
    write_file_ID,
    flagplot,
    rarray=[],
    azarray=[],
    nt=0,
):
    ###############################################################
    # PARAMETERS FOR FUTURE FUNCTIONALITY
    ###################################################################
    tukey_window = 2
    extra_thresh = []

    ###############################################################
    # START TRACKING!!
    ###################################################################

    # Setup some initial arrays and check inputs.

    newumat = 0
    newvmat = 0
    wasarray = 0 * StormLabels  # set up array of zeros.
    lifearray = 0 * StormLabels
    numstorms = StormLabels.max()
    print("numstorms = ", numstorms)
    StormData = []

    # Check xmat and ymat have same shape
    np.testing.assert_equal(
        xmat.shape, ymat.shape, "xmat and ymat must have same shape"
    )

    # Check squarehalf input for validity. If it is too large compared to domain size, raise exception
    min_domain_size = np.min((xmat.shape[0], xmat.shape[1]))
    if squarehalf * 2 >= min_domain_size:
        raise ValueError(
            "squarehalf input is too large for domain size. Reduce squarehalf."
        )

    # Check if this is the first timestep using OldStormData variable.
    # If it is, initialise this first field and return
    # TODO: check how much of this is necessary!!
    if len(OldStormData) == 0:
        return initialise_first_field(
            field,
            newwas,
            StormLabels,
            xmat,
            ymat,
            squarehalf,
            num_dt,
            misval,
            doradar,
            under_threshold,
            rarray,
            azarray,
        )

    # Check if there are no storms in either time step
    if np.max(OldStormLabels) <= 0 and np.max(StormLabels) <= 0:
        print("No storms to track in either time step. Returning empty arrays.")
        return StormData, newwas, StormLabels, newumat, newvmat, wasarray, lifearray

    ###################################################
    # OldStormData & StormData ARE NOT EMPTY, SO USE FFT TO GET VELOCITIES
    # AND UPDATE UVLABEL IN OldStormData ACCORDINGLY
    # Estimate velocities using squares within domain
    ###################################################

    # xint, yint: bounds of the squares used to calculate fft??
    # xint and yint shape is the number of squarehalfs in xmat and ymat
    xint, yint = np.meshgrid(
        range(xmat[0, 0] + squarehalf, xmat[0, -1], squarehalf),
        range(ymat[0, 0] + squarehalf, ymat[-1, 0], squarehalf),
    )

    # init wind vector arrays??
    buu = np.full(xint.shape, np.nan)
    bvv = np.full(xint.shape, np.nan)
    bww = np.full(xint.shape, np.nan)

    # TODO: corx is actually the 1 dimension, not the 0 dimension
    # TODO: can probably do some sort of permutation to get these iterables
    # rather than nested for loops.
    for corx in range(0, xint.shape[0]):
        if flagplot:
            nij = -3
            # fig, axs = plt.subplots(np.size(xint,1),3, figsize=(6,2*np.size(xint,1)), facecolor='w', edgecolor='k')
            fig, axs = plt.subplots(
                int(0.5 * np.size(xint, 1)) + 1,
                6,
                figsize=(6, np.size(xint, 1)),
                facecolor="w",
                edgecolor="k",
            )
            axs = axs.ravel()

        for cory in range(0, xint.shape[1]):
            if flagplot:
                nij = nij + 3
                # What is this logic??

            oldsquare = prev_features[
                (squarehalf) * corx : (squarehalf) * corx + 2 * squarehalf,
                (squarehalf) * cory : (squarehalf) * cory + 2 * squarehalf,
            ]
            newsquare = current_features[
                (squarehalf) * corx : (squarehalf) * corx + 2 * squarehalf,
                (squarehalf) * cory : (squarehalf) * cory + 2 * squarehalf,
            ]

            # if there are too few storms, don't try to derive motion vectors.
            if np.sum(oldsquare) < fftpixels or np.sum(newsquare) < fftpixels:
                buu[corx, cory] = np.nan
                bvv[corx, cory] = np.nan
                bww[corx, cory] = np.nan

            else:
                # So, dx and dy are the indices of maximum fft motion field
                # amp is the maximum value normalised by... a certain pre-fft value (cartesian distance?)
                # ffv is the full fft motion field
                dx, dy, amplitude, corrval = ffttrack(
                    oldsquare, newsquare, tukey_window
                )

                ## indices are upside down so need minus to get real-world dy-velocity
                buu[corx, cory] = dx
                bvv[corx, cory] = dy
                # bww[corx, cory] = amplitude
                if flagplot:
                    # print(corx)
                    # print(cory)
                    # print(nij)
                    axs[nij].pcolormesh(oldsquare)
                    axs[nij].set_title(str(int(np.sum(oldsquare))))
                    axs[nij + 1].pcolormesh(newsquare)
                    axs[nij + 1].set_title(str(int(np.sum(newsquare))))
                    # axs[nij + 2].pcolormesh(corrval)
                    axs[nij + 2].set_title("(" + str(dx) + "," + str(dy) + ")")

        if flagplot:
            plt.savefig(
                IMAGES_DIR + "Correlations_" + write_file_ID + "_" + str(corx) + ".png"
            )
            plt.close()

    old_filter_method = False

    if old_filter_method:
        include_missing_case = True
        # CHECK NEIGHBOURING VALUES FOR SMOOTHNESS
        # Ignore warnings about mean over empty array in this section
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Loop over all subdomains again, this time with knowledge of where to find
            # the dx and dy motion vectors
            # Uses the dx and dy indices to check neighbouring values
            # with any values on boundaries handled differently
            # So, here, corx and cory are subdomain locations
            for corx in range(0, int(np.size(xint, 0))):
                for cory in range(0, int(np.size(xint, 1))):
                    bu_nb = np.nan
                    bv_nb = np.nan
                    if np.isnan(buu[corx, cory]) and np.isnan(bvv[corx, cory]):
                        continue
                    if corx == 0:
                        if cory == 0:
                            # Lower left corner
                            bu_nb = np.nanmean([buu[0, 1], buu[1, 0], buu[1, 1]])
                            bv_nb = np.nanmean([bvv[0, 1], bvv[1, 0], bvv[1, 1]])
                        elif cory == int(np.size(xint, 1)) - 1:
                            # lower right corner
                            bu_nb = np.nanmean(
                                [buu[0, cory - 1], buu[1, cory], buu[1, cory - 1]]
                            )
                            bv_nb = np.nanmean(
                                [bvv[0, cory - 1], bvv[1, cory], bvv[1, cory - 1]]
                            )
                        else:
                            # Lower boundary, not corners
                            bu_nb = np.nanmean(
                                [
                                    buu[0, cory + 1],
                                    buu[0, cory - 1],
                                    buu[1, cory - 1],
                                    buu[1, cory],
                                    buu[1, cory + 1],
                                ]
                            )
                            bv_nb = np.nanmean(
                                [
                                    bvv[0, cory + 1],
                                    bvv[0, cory - 1],
                                    bvv[1, cory - 1],
                                    bvv[1, cory],
                                    bvv[1, cory + 1],
                                ]
                            )
                    elif corx == int(np.size(xint, 0)) - 1:
                        if cory == 0:
                            bu_nb = np.nanmean(
                                [buu[corx, 1], buu[corx - 1, 0], buu[corx - 1, 1]]
                            )
                            bv_nb = np.nanmean(
                                [bvv[corx, 1], bvv[corx - 1, 0], bvv[corx - 1, 1]]
                            )
                        elif cory == int(np.size(xint, 1)) - 1:
                            bu_nb = np.nanmean(
                                [
                                    buu[corx, cory - 1],
                                    buu[corx - 1, cory],
                                    buu[corx - 1, cory - 1],
                                ]
                            )
                            bv_nb = np.nanmean(
                                [
                                    bvv[corx, cory - 1],
                                    bvv[corx - 1, cory],
                                    bvv[corx - 1, cory - 1],
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
                    elif include_missing_case and cory == 0:
                        # TODO: Added in possibly missing case?
                        bu_nb = np.nanmean(
                            [
                                buu[corx, cory + 1],
                                buu[corx - 1, cory],
                                buu[corx - 1, cory + 1],
                                buu[corx + 1, cory + 1],
                                buu[corx + 1, cory],
                            ]
                        )
                        bv_nb = np.nanmean(
                            [
                                bvv[corx, cory + 1],
                                bvv[corx - 1, cory],
                                bvv[corx - 1, cory + 1],
                                bvv[corx + 1, cory + 1],
                                bvv[corx + 1, cory],
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
                    if np.abs(buu[corx, cory] - bu_nb) > dd_tolerance * num_dt:
                        buu[corx, cory] = np.nan
                    if np.abs(bvv[corx, cory] - bv_nb) > dd_tolerance * num_dt:
                        bvv[corx, cory] = np.nan

    else:
        # footprint excludes current index.
        footprint = np.ones((3, 3))
        footprint[1, 1] = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bu_nb = ndimage.generic_filter(
                buu, np.nanmean, footprint=footprint, mode="constant", cval=np.nan
            )
            bv_nb = ndimage.generic_filter(
                bvv, np.nanmean, footprint=footprint, mode="constant", cval=np.nan
            )
        buu[np.abs(buu - bu_nb) > dd_tolerance * num_dt] = np.nan
        bvv[np.abs(bvv - bv_nb) > dd_tolerance * num_dt] = np.nan

    ## ACTUAL DISPLACEMENT
    # Interpolate these displacements onto the full grid
    newumat = interpolate_speeds(xint, yint, xmat, ymat, buu)
    newvmat = interpolate_speeds(xint, yint, xmat, ymat, bvv)
    # print(f"xmat_shape: {xmat.shape}")
    # print(f"field shape: {field.shape}")

    # newumat = interpolate_subdomain_flows(yint, xint, buu, xmat.shape)
    # newvmat = interpolate_subdomain_flows(yint, xint, bvv, xmat.shape)

    with open(f"{IMAGES_DIR}/umat_{nt}.npy", "xb") as f:
        np.save(f, newumat)

    with open(f"{IMAGES_DIR}/vmat_{nt}.npy", "xb") as f:
        np.save(f, newvmat)

    # Assign displacement to each of the old storms.

    # Setup a field of new labels
    newlabel = np.zeros(OldStormLabels.shape)

    # Loop over all OldStormData to populate the newlabel array
    for ns in range(len(OldStormData)):
        # get the label for the storm
        jj = OldStormData[ns].storm
        # Find idxs of this storm in old data
        labelind = np.where(OldStormLabels == jj)
        # Use these idxs to get dx and dy
        # TODO: is this the most sensible way of doing this?
        # E.g., storms that are between subdomain boundaries might advect in slightly
        # different ways, but this doesn't really acknowldge the spatial heteogenity of
        # wind field. Instead, it advects all pixels by the same amount...
        dx = np.mean(newumat[labelind])
        dy = np.mean(newvmat[labelind])
        # If dx and dy == 0, newlabel locations are provisionally set
        # to the old one
        if dx == 0.0 and dy == 0.0:
            newlabel[labelind] = jj
        else:
            # Loop over all coords in the Feature, where ii is an index
            # for the coord in the list labelind, not the coord itself
            for ii in range(np.size(labelind, 1)):
                # Ummm... this seems like the wrong way around???
                # newyindex has dx added rather than dy?
                # Probably a quick fix for the row-ordered np approach and inconsistent
                # definitions of yx and dy dx.
                newyind = labelind[1][ii] + int(np.around(dx))
                newxind = labelind[0][ii] + int(np.around(dy))
                # Check for Feature coord going OOB?
                # If it does, no further investigation in this loop
                if (
                    newxind > np.size(newlabel, 0) - 1
                    or newyind > np.size(newlabel, 1) - 1
                    or newxind < 0
                    or newyind < 0
                ):
                    continue
                # If there is already a label assigned to this coord from a previous Feature
                elif newlabel[newxind, newyind] > 0:
                    # Decide whether to replace the existing label with the current label
                    # Oh, nq is 1 - the label that is currently at the coord
                    # and ns is the current label
                    # TODO: But why is this looking at 1 - label? why is this useful?
                    # Presumbaly this is because this label is 0 ordered or something??
                    nq = int(newlabel[newxind, newyind] - 1)
                    # Check whether distance between the current iterating label, or the exisitng
                    # label is closer to the older data?
                    # This logic does theoretically mean that a pixel/coord in the new frame may
                    # be assigned to a different feature than all other pixels in this ns loop
                    olddist = (
                        xmat[newxind, newyind] - OldStormData[nq].centroidx
                    ) ** 2 + (ymat[newxind, newyind] - OldStormData[nq].centroidy) ** 2
                    newdist = (
                        xmat[newxind, newyind] - OldStormData[ns].centroidx
                    ) ** 2 + (ymat[newxind, newyind] - OldStormData[ns].centroidy) ** 2
                    # If the new label is closer, replace the coordinate
                    if newdist < olddist:
                        newlabel[newxind, newyind] = jj
                else:
                    newlabel[newxind, newyind] = jj

    # So, now, newlabel or QuvL is the list of advected labels
    # Now, populate AdvectedStorms array with new labels and sizes of the Feature
    # which may have changed given the above logic.
    # TODO: this is important to be aware of in my code.
    QuvL = newlabel
    AdvectedStorms = np.zeros([len(OldStormData), 3])
    for ns in range(len(OldStormData)):
        jj = OldStormData[ns].storm
        centrind = np.where(QuvL == jj)
        if np.size(centrind, 1) == 0:
            continue
        else:
            AdvectedStorms[ns][0] = np.mean(xmat[centrind])
            AdvectedStorms[ns][1] = np.mean(ymat[centrind])
            # Only need to check one axis, since np.where produces a tuple of x,y coords
            # so the 1 axis is the same as he 0 axis, and size is looking at the number of
            # grid points included in the storm.
            AdvectedStorms[ns][2] = int(np.size(centrind, 1))

    ###################################################
    # NOW LOOP THROUGH StormData AND CHECK FOR OVERLAP WITH
    # ADVECTED OldStormData STORMS
    ###################################################
    # Construct some containing arrays TODO: figure out what these are
    wasnum = np.zeros(len(StormData))  # Same size as current storm data
    qbins = range(int(np.max(OldStormLabels)) + 2)  # from 0 to max(OldStormData + 2)
    qarea = np.ones(
        [int(np.max(OldStormLabels)) + 1]
    )  # from 0 to max(OldStormData + 1)
    qlife = np.ones(
        [int(np.max(OldStormLabels)) + 1]
    )  # from 0 to max(OldStormData + 1)

    # Loop over all OLD storms
    for qq in range(len(OldStormData)):
        # If size of advected storm (no of pixels) > 0
        # Then add this area to the qarea array
        # TODO: qarea is set up as np.ones, so if size == 0, then its area will actually be set to 1.
        # Is this intended??
        if AdvectedStorms[qq, 2] > 0:
            qarea[qq + 1] = AdvectedStorms[qq, 2]
        # Add the lifetime from the old storm to this lifetime array
        qlife[qq + 1] = OldStormData[qq].life

    # Loop over IDXS for the current storms
    for ns in range(numstorms):
        jj = ns + 1  # first storm is labelled 1, but python indeces start at 0.
        # 'C' are the tuple of coords with this storm label
        C = np.where(StormLabels == jj)
        # StormData is just an empty list until the first iteration
        StormData += [
            StormS(
                jj,
                StormLabels,
                field,
                xmat,
                ymat,
                newwas,
                newumat,
                newvmat,
                num_dt,
                misval,
                doradar,
                under_threshold,
                extra_thresh=extra_thresh,
                storm_history=True,
                string=None,
                rarray=rarray,
                azarray=azarray,
            )
        ]
        # These arrays also just contain 0 until they are filled here
        # They are the same size as input field
        # It seems like wasarray is just storm label field again??? We'll see.
        # Lifetime array shows the number of conseuctive timesteps that
        # a feature was present at that pixel
        wasarray[C] = int(jj)
        lifearray[C] = 1

        ###################################################
        # CHECK OVERLAP WITH QHIST
        # IF NO OVERLAP, THEN
        # GENERATE (halo) km RADIUS AROUND CENTROID
        # CHECK FOR OVERLAP WITHIN (halo) km OF CENTROID
        ###################################################
        qhist = (np.histogram(QuvL[np.where(StormLabels == jj)], qbins))[0][:] / float(
            StormData[ns].area
        ) + (np.histogram(QuvL[np.where(StormLabels == jj)], qbins))[0][:] / qarea[:]
        # if nt==35 and jj==131:
        #   raise ValueError("Check storm 131 (or 120)")

        # Then check for overlaps by looking at MAX from idx 1 onwards (i.e., ignore 0 since this is the fill value
        # TODO: this check does not guarantee that the overlap is being compared with the correct label
        # I.e., could there be a situation where there is not enough overlap in the expected label,
        # but there is in a different label?
        # this might be okay if it will be handled in the next section of code.
        # All this cares about is whether there is at least ONE label with sufficient overlap
        if np.max(qhist[1:]) < lapthresh:
            # If the overlap isn't met, then construct a new mask that instead is a halo/nbhood around the advected storm
            # to use as the histogram for future use.
            newblob = 0 * xmat
            blobind = np.where(
                (xmat - StormData[ns].centroidx) ** 2
                + (ymat - StormData[ns].centroidy) ** 2
                < halosq
            )
            newblob[blobind] = newblob[blobind] + 1
            qhist = (np.histogram(QuvL[np.where(newblob == 1)], qbins))[0][:] / float(
                StormData[ns].area
            ) + (np.histogram(QuvL[np.where(newblob == 1)], qbins))[0][:] / qarea[:]

        ###################################################
        # IF OVERLAP, THEN
        # - INHERIT "WAS"
        # - UPDATE "LIFE" AND "TRACK" AND "WASDIST"
        # - INHERIT "dx" AND "dy" (ONLY UPDATE IF SINGLE OVERLAP)
        ###################################################
        # If there is at least one label that contains sufficient overlap between advected and current storms, then
        # enter this loop.
        if np.max(qhist[1:]) >= lapthresh:
            # Get the number of overlaps that satisfy the overlap condition
            numlaps = np.where(qhist[1:] >= lapthresh)
            ###################################################
            # IF MORE THAN ONE GOOD OVERLAP
            # KEEP PROPERTIES OF STORM WITH LARGEST OVERLAP
            # IF MORE THAN ONE LARGEST, KEEP NEAREST IN CENTROID
            ###################################################
            if np.size(numlaps, 1) > 1:
                # Setup arrays with size of the number of good overlaps
                lapdist = np.zeros([np.size(numlaps, 1)])
                sectlap = np.zeros([np.size(numlaps, 1)])

                # Loop from 0 -> number of overlaps - 1
                for kkind in range(np.size(numlaps, 1)):
                    # Get the label for this iterator
                    qindex = np.squeeze(numlaps[0][kkind])
                    # Look at centroid
                    lapdist[kkind] = np.sqrt(
                        (StormData[ns].centroidx - AdvectedStorms[qindex, 0]) ** 2
                        + (StormData[ns].centroidy - AdvectedStorms[qindex, 1]) ** 2
                    )
                    # look at overlap
                    sectlap[kkind] = np.size(
                        np.where((QuvL == qindex + 1) & (StormLabels == jj)), 1
                    )

                # get label of largest overlap
                kmax = np.where(sectlap == np.max(sectlap))

                # If there are multiple labels with equal overlap
                if np.size(kmax, 1) > 1:
                    # This gets the minimum centroid distance from current label
                    kkmax = kmax[0][
                        np.where(lapdist[kmax[0][:]] == np.min(lapdist[kmax[0][:]]))
                    ]
                    # If the above still doesn't find a storm, then just choose the first one.
                    if np.size(kkmax) > 1:
                        kkmax = kmax[0][kkmax[0]]
                # Otherwise, there is only one label with largest overlap, so use this.
                else:
                    kkmax = kmax[0][0]

                # idx of the label to use as largest overlap for assigning to this storm.
                # Update the current storm data object with these details, then go to merging/splitting
                kindex = np.squeeze(numlaps[0][kkmax])
                StormData[ns].inherit_properties(
                    jj,
                    OldStormData,
                    kindex,
                    QuvL,
                    StormLabels,
                    qhist,
                    lapthresh,
                    misval,
                    single_overlap=False,
                )
                # Update these arrays
                wasarray[C] = OldStormData[kindex].was
                lifearray[C] = StormData[ns].life

            ###################################################
            # SINGLE OVERLAP
            ###################################################
            else:
                zindex = np.squeeze(numlaps[0][0])
                # lapdist = np.sqrt(
                #     (StormData[ns].centroidx - OldStormData[zindex].centroidx) ** 2
                #     + (StormData[ns].centroidy - OldStormData[zindex].centroidy) ** 2
                # )
                StormData[ns].inherit_properties(
                    jj,
                    OldStormData,
                    zindex,
                    QuvL,
                    StormLabels,
                    qhist,
                    lapthresh,
                    misval,
                    single_overlap=True,
                )
                wasarray[C] = OldStormData[zindex].was
                lifearray[C] = StormData[ns].life

        ###################################################
        # IF NO OVERLAP, THEN (NEW STORM)
        # - "WAS" SET TO CURRENT MAX LABEL +1
        # - UPDATE "LIFE" AND "TRACK" AND "WASDIST" FOR A NEW STORM
        ###################################################
        else:
            StormData[ns].was = newwas
            wasarray[C] = newwas
            StormData[ns].life = 1
            lifearray[C] = 1
            newwas = newwas + 1

    wasnum = np.array([StormData[ns].was for ns in range(len(StormData))])
    ###################################################
    # QUICK SANITY CHECK
    # ACCRETED SHOULD NEVER BE A VALUE
    # SIMILAR TO EXISTING STORM ID
    ###################################################
    for ns in range(len(StormData)):
        jj = StormData[ns].storm
        if StormData[ns].accreted[-1] == misval:
            continue
        else:
            for acnum in range(np.size(StormData[ns].accreted)):
                acind = np.where((wasnum - StormData[ns].accreted[acnum]) == 0)
                if np.size(acind, 1) > 0:
                    StormData[ns].accreted[acnum] = misval
            # acnew=np.where(StormData[ns].accreted > misval)
            acnew = [aci for aci in StormData[ns].accreted if aci > misval]
            if np.size(acnew) > 0:
                for acindex in range(np.size(acnew)):
                    StormData[ns].accreted[acnum] = acnew[acindex]
            else:
                StormData[ns].accreted = [misval]

    ###################################################
    # TRACKING MERGING BREAKING
    # MULTIPLE STORMS AT T (StormData) MAY HAVE SAME LABEL "WAS"
    # FIND STORM WITH LARGEST OVERLAP AT T+1 WITH ADVECTED q(T)
    # THIS IS THE "PARENT" STORM,
    # "PARENT" VECTOR WITH INDICES OF NEW LABELS FOR "CHILD" STORMS
    # STORMS WITH SAME WAS BUT FUTHER FROM CENTROID ARE "CHILD", VALUE "PARENT"
    ###################################################

    # Loop over all storms
    for ns in range(len(StormData)):
        jj = StormData[ns].storm

        # wasdist = np.size(np.where((QuvL == kindex + 1) & (StormLabels == jj)), 1)
        # i.e., size of exact overlap between advected feature and current feature
        # initialised as default misval if it has not inherited properties, ie, is a new storm
        if StormData[ns].wasdist == [misval]:
            continue

        # get idxs of all storms in wasnum with the same value as current value
        # wasnum = np.array([StormData[ns].was for ns in range(len(StormData))])
        wasind = np.where(wasnum == wasnum[ns])
        wasseplength = 0
        for kkind in range(np.size(wasind)):
            if StormData[wasind[0][kkind]].wasdist == [misval]:
                continue
            else:
                wasseplength = wasseplength + 1

        wassep = np.zeros(wasseplength)
        if np.size(wasind) > 1:
            kkval = 0
            for kkind in range(np.size(wasind)):
                if StormData[wasind[0][kkind]].wasdist == [misval]:
                    continue
                else:
                    wassep[kkval] = StormData[wasind[0][kkind]].wasdist
                    kkval = kkval + 1
        else:
            wassep = StormData[wasind[0][0]].wasdist

        ########################################
        # WASSEP NOW CONTAINS ALL NON-ZERO OVERLAP VALUES
        # FIND THE MAXIMUM (THIS WILL BE THE PARENT)
        # ALL OTHER STORMS WILL BE THE CHILDREN
        #########################################
        if isinstance(wassep, (int, float)):
            wassep = np.array([wassep])
        kmax = np.where(wassep == np.max(wassep))
        kkmax = np.min(kmax)
        children = []
        # This bit finds children that have spawned, gives them a new id...
        for kkind in range(np.size(wassep)):
            if not kkind == kkmax:
                StormData[wasind[0][kkind]].child = StormData[wasind[0][kkmax]].was
                StormData[wasind[0][kkind]].was = newwas
                wasarray[np.where(StormLabels == wasind[0][kkind] + 1)] = newwas
                StormData[wasind[0][kkind]].life = StormData[wasind[0][kkmax]].life
                lifearray[np.where(StormLabels == wasind[0][kkind] + 1)] = StormData[
                    wasind[0][kkmax]
                ].life
                newwas = newwas + 1
                wasnum[wasind[0][kkind]] = StormData[wasind[0][kkind]].was
                children.append(StormData[wasind[0][kkind]].was)
                StormData[wasind[0][kkind]].wasdist = misval

        ###################################################
        # UPDATE PARENT STORM WITH CHILDREN
        ###################################################
        if np.size(children) > 0:
            StormData[wasind[0][kkmax]].parent = children

    return StormData, newwas, StormLabels, newumat, newvmat, wasarray, lifearray


###################################################
# interpolate_speeds used for (dx,dy) calculation where no objects are identified.
###################################################


def interpolate_speeds(xint, yint, xmat, ymat, buu):
    valid_mask = ~np.isnan(buu)
    coords = np.array(np.nonzero(valid_mask)).T
    values = buu[valid_mask]
    if np.size(values) >= 4:
        it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
        print(f"it: {it}")
        filled = it(list(np.ndindex(buu.shape))).reshape(buu.shape)
        print(f"filled: {filled}")

        # interp2d deprecated in newer version of scipy.
        # For functionally identical replacement, use RectBivariateSpline
        # with kx=3, ky=3 for cubic spline interpolation, and additional transposing.
        # https://scipy.github.io/devdocs/tutorial/interpolate/interp_transition_guide.html
        # fu = interpolate.interp2d(xint[0, :], yint[:, 0], filled, kind="cubic")
        fu = interpolate.RectBivariateSpline(
            xint[0, :], yint[:, 0], filled.T, kx=3, ky=3
        )
        newumat = fu(xmat[0, :], ymat[:, 0]).T
    else:
        newumat = np.zeros(np.shape(OldStormLabels))
    return newumat


def interpolate_subdomain_flows(
    y_subdomain_bounds, x_subdomain_bounds, subdomain_flows, full_domain_shape
) -> NDArray:
    valid_mask = ~np.isnan(subdomain_flows)
    coords = np.array(np.nonzero(valid_mask)).T
    values = subdomain_flows[valid_mask]
    xmat, ymat = np.meshgrid(
        range(int(full_domain_shape[1] / -2), int(full_domain_shape[1] / 2)),
        range(int(full_domain_shape[0] / -2), int(full_domain_shape[0] / 2)),
    )
    # TODO: the below doesn't work becuase this is then on a different
    # coord system to x/y_subdomain_bounds. Would need to shift these to be
    # on the same coord system to get the correct flow.
    # xmat, ymat = np.meshgrid(range(full_domain_shape[0]), full_domain_shape[1])
    # print(f"xmat inside function shape: {xmat.shape}")
    if np.size(values) >= 4:
        it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
        filled = it(list(np.ndindex(subdomain_flows.shape))).reshape(
            subdomain_flows.shape
        )

        # interp2d deprecated in newer version of scipy.
        # For functionally identical replacement, use RectBivariateSpline
        # with kx=3, ky=3 for cubic spline interpolation, and additional transposing.
        # https://scipy.github.io/devdocs/tutorial/interpolate/interp_transition_guide.html
        # fu = interpolate.interp2d(xint[0, :], yint[:, 0], filled, kind="cubic")
        fu = interpolate.RectBivariateSpline(
            x_subdomain_bounds[0, :], y_subdomain_bounds[:, 0], filled.T, kx=3, ky=3
        )
        newumat = fu(xmat[0, :], ymat[:, 0]).T
    else:
        newumat = np.zeros(full_domain_shape)
    return newumat


###################################################
# label_storms IS A FLOOD FILL ALGORITHM RETURNING
# AN ARRAY OF UNIQUELY LABELLED STORMS
# THIS FUNCTION IS ESSENTIAL AND SHOULD ONLY BE ALTERED
# BY EXPERIENCED USERS
###################################################


def label_storms(
    field: NDArray[np.floating],
    min_area: float,
    threshold: float,
    under_threshold: bool = False,
    connectivity_structure: NDArray[np.bool] = np.ones((3, 3)),
) -> NDArray[np.integer]:
    """
    Label distinct regions in the input field that meet a specified threshold condition.

    Args:
        field (np.ndarray):
            2D input array of data to be labelled
        min_area (float):
            Minimum area (in number of grid points) for a region to be considered valid
        threshold (float):
            Threshold value for identifying regions
        under_threshold (bool, optional):
            If True, regions under the threshold are considered;
            if False, regions over the threshold are considered.
            Defaults to False.
        connectivity_structure (NDArray, optional):
            Boolean array defining connectivity for region labelling.
            Default is 8-way connectivity, meaning all cardinal AND diagonal neighbours that
            meet the threshold condition are considered part of the same region.
            An alternative arrangement would be 4-way connectivity (diagonals omitted), defined as:
            np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])
            See scipy.ndimage.label documentation for more details.
            Defaults to np.ones((3, 3)).

    Raises:
        TypeError: field must be a numpy ndarray
        ValueError: min_area must be a non-negative number
        ValueError: threshold must be a number"
        TypeError: under_threshold must be a boolean
        ValueError: field must be a 2D array

    Returns:
        NDArray[np.int_]: 2D Integer field of labelled regions, same shape as input field
    """

    # Check input types
    if not isinstance(field, np.ndarray):
        raise TypeError("field must be a numpy ndarray")
    if not isinstance(min_area, (int, float)) or min_area < 0:
        raise ValueError("min_area must be a non-negative number")
    if not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a number")
    if not isinstance(under_threshold, bool):
        raise TypeError(
            f"under_threshold must be a boolean, got {type(under_threshold)}"
        )

    # Check the input field is 2D
    if field.ndim != 2:
        raise ValueError("field must be a 2D array")

    # Construct feature field using threshold and threshold condition
    # Grid points meeting the condition are set to 1, others to 0
    if under_threshold:
        feature_field = np.where(field < threshold, 1, 0)
    else:
        feature_field = np.where(field > threshold, 1, 0)

    # Identify and label distinct regions in the feature field
    id_regions, num_ids = ndimage.label(feature_field, structure=connectivity_structure)

    # Any regions smaller than the min_area are removed from the feature field
    # before re-running feature labelling
    id_sizes = np.array(ndimage.sum(feature_field, id_regions, range(num_ids + 1)))
    area_mask = id_sizes < min_area
    feature_field[area_mask[id_regions]] = 0
    id_regions, num_ids = ndimage.label(feature_field, structure=connectivity_structure)
    print("num_ids = ", num_ids)

    return id_regions


##############################################################
# ffttrack
##############################################################
# [dx, dy, amp] = ffttrack(s1, s2, method)
# Input:
# s1 = oldsquare
# s2 = newsquare
# method = 1 for TUKEY WINDOW (TAPERED COSINE)
# Output:
# dx = distance in x-direction from previous cell
# dy = distance in y-direction from previous cell
# amp = amplitude
# ffv = full output, only needed for testing (if plotting with flagplot)
##############################################################


def ffttrack(s1, s2, method):
    leno = max(np.size(s1, 0), np.size(s1, 1))

    if method == 1:
        alpha = max(0.1, 10.0 / leno)
        xhan = np.array(np.arange(0.5, leno + 0.5))
        hann1 = np.ones([np.size(xhan)])
        hann1[np.where(xhan < alpha * leno / 2.0)] = 0.5 * (
            1
            + np.cos(
                np.pi
                * (2 * xhan[np.where(xhan < alpha * leno / 2.0)] / (alpha * leno) - 1)
            )
        )
        hann1[np.where(xhan > leno * (1 - alpha / 2.0))] = 0.5 * (
            1
            + np.cos(
                np.pi
                * (
                    2 * xhan[np.where(xhan > leno * (1 - alpha / 2.0))] / (alpha * leno)
                    - 2.0 / alpha
                    + 1
                )
            )
        )
        hann2 = hann1.conj().transpose() * hann1
    elif method == 2:
        hann2 = hann(leno)
    else:
        xhan = np.array(np.arange(0.5, leno + 0.5))
        hann1 = np.ones([np.size(xhan)])
        hann2 = hann1.conj().transpose() * hann1

    ## FIND CONVOLUTION S1, S2 USING FFT

    b1 = s1 * hann2
    b2 = s2 * hann2

    m1 = b1 - np.mean(b1)
    m2 = b2 - np.mean(b2)

    normval = np.sqrt(np.sum(m1**2) * np.sum(m2**2))

    # # ffv = signal.fftconvolve(s1,s2,mode='same')
    # ffv = np.real(np.fft.ifft2(np.fft.fft2(m2) * (np.fft.fft2(m1)).conj()))

    phase_corr = phase_cross_correlation(
        m1,
        m2,
        space="real",
        overlap_ratio=0.6,
        normalization=None,
        upsample_factor=1,
        disambiguate=False,
    )
    # Note: dy, dx are vectors going from m2 -> m1, so multiply -1
    dy, dx = phase_corr[0] * -1

    return dx, dy, 0, 0

    val = np.max(ffv)
    ind = np.where(ffv == val)

    # print 'max ffv and ind -> ',val, ind
    dx = ind[1][0]
    dy = ind[0][0]

    ## 1hour -> 25km(leno/2) ; 5mins -> 2km(leno/10) : 10mins -> 4km(leno/5)
    cv = leno / 2  # Org. from Thorld = 25km
    # cv = leno/2 # For 200m grids = 20km
    if dx > cv:
        dx = dx - leno  # Org. from Thorld
        # dx = dx - (cv*(dx/cv))
    if dy > cv:
        dy = dy - leno  # Org. from Thorld
        # dy = dy - (cv*(dy/cv))
    amp = val / normval

    return dx, dy, amp, ffv


###################################################
# write_storms produces TXT file for analysis of tracked object properties
###################################################


def write_storms(
    file_ID,
    init_time,
    now_time,
    label_method,
    squarelength,
    rafraction,
    newwas,
    StormData,
    doradar,
    misval,
    IMAGES_DIR,
):
    if not Path(IMAGES_DIR).exists():
        Path(IMAGES_DIR).mkdir(exist_ok=True)
    # print("IMAGES_DIR + file_ID +'.txt'=", IMAGES_DIR + file_ID +'.txt')
    fw = open(IMAGES_DIR + "history_" + file_ID + ".txt", "w")
    fw.write("missing_value=" + str(misval) + "\r\n")
    fw.write("Start date and time=" + init_time.strftime("%d/%m/%y-%H%M") + "\r\n")
    fw.write("Current date and time=" + now_time.strftime("%d/%m/%y-%H%M") + "\r\n")
    fw.write("Label method=" + label_method + "\r\n")
    fw.write("Squarelength=" + str(squarelength) + "\r\n")
    fw.write("Rafraction=" + str(rafraction) + "\r\n")
    fw.write("total number of tracked storms=" + str(newwas - 1) + "\r\n")
    for ns in range(len(StormData)):
        fw.write("storm " + str(StormData[ns].was))
        #       fw.write(' label=' + str(StormData[ns].storm)) # Matches storm to label in mask. Actually no need for this as it is the same as it matches the order of the storms.
        fw.write(" area=" + str(StormData[ns].area))
        fw.write(
            " centroid="
            + str(round(StormData[ns].centroidx, 2))
            + ","
            + str(round(StormData[ns].centroidy, 2))
        )
        fw.write(
            " box="
            + str(StormData[ns].boxleft)
            + ","
            + str(StormData[ns].boxup)
            + ","
            + str(StormData[ns].boxwidth)
            + ","
            + str(StormData[ns].boxheight)
        )
        fw.write(" life=" + str(StormData[ns].life))
        fw.write(
            " dx="
            + str(round(StormData[ns].dx, 2))
            + " dy="
            + str(round(StormData[ns].dy, 2))
        )

        if doradar:
            fw.write(
                " range="
                + str(round(StormData[ns].rangel, 2))
                + ","
                + str(round(StormData[ns].rangeu, 2))
            )
            fw.write(
                " azimuth="
                + str(round(StormData[ns].azimuthl, 2))
                + ","
                + str(round(StormData[ns].azimuthu, 2))
            )
        fw.write(" meanv=" + str(round(StormData[ns].meanvar, 2)))
        fw.write(" extreme=" + str(round(StormData[ns].extreme, 2)) + " accreted=")
        if np.size(StormData[ns].accreted) > 1:
            for acind in range(np.size(StormData[ns].accreted) - 1):
                fw.write(str(StormData[ns].accreted[acind]) + ",")
            fw.write(
                str(StormData[ns].accreted[np.size(StormData[ns].accreted) - 1])
                + " parent="
            )
        else:
            fw.write(str(StormData[ns].accreted[-1]) + " parent=")
        fw.write(str(StormData[ns].child) + " child=")
        if np.size(StormData[ns].parent) > 1:
            for acind in range(np.size(StormData[ns].parent) - 1):
                fw.write(str(StormData[ns].parent[acind]) + ",")
            fw.write(
                str(StormData[ns].parent[np.size(StormData[ns].parent) - 1]) + "\r\n"
            )
        else:
            fw.write(str(StormData[ns].parent[-1]) + "\r\n")
    fw.close()
