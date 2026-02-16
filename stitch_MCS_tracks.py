'''
Created on Feb 27 2024

@author: wkeat
'''


import os
import _pickle as pickle
from classes import MISVAL, Track
import datetime as dt
import sys
import numpy as np
import cftime
import xarray as xr
import iris
import pandas as pd
from datetime import datetime, timedelta


def generate_360_day_range(start, end):
    """
    Generate a date range manually for a 360-day calendar.
    """
    dates = []
    current_date = start

    while current_date <= end:
        dates.append(current_date)

        # Manually increment day, month, and year
        if current_date.day < 30:  # Increment day within a month
            next_day = current_date.day + 1
            next_month = current_date.month
            next_year = current_date.year
        elif current_date.month < 12:  # Move to the next month
            next_day = 1
            next_month = current_date.month + 1
            next_year = current_date.year
        else:  # Move to the next year
            next_day = 1
            next_month = 1
            next_year = current_date.year + 1

        # Update current_date
        current_date = cftime.Datetime360Day(next_year, next_month, next_day)

    return dates

def generate_360_hour_range(start, end, increment_minutes=60):
    """
    Generate a date range for a 360-day calendar including hours and minutes.
    Manually increments by a specified number of minutes.
    """
    dates = []
    current_date = start

    while current_date <= end:
        dates.append(current_date)
        # Increment the time
        minute = current_date.minute + increment_minutes
        hour = current_date.hour
        day = current_date.day
        month = current_date.month
        year = current_date.year

        # Handle minute overflow
        if minute >= 60:
            minute -= 60
            hour += 1

        # Handle hour overflow
        if hour >= 24:
            hour -= 24
            day += 1

        # Handle day overflow
        if day > 30:  # Each month has 30 days in a 360-day calendar
            day = 1
            month += 1

        # Handle month overflow
        if month > 12:  # Each year has 12 months
            month = 1
            year += 1

        # Update current_date
        current_date = cftime.Datetime360Day(year, month, day, hour, minute)

    return dates



def rim_remove(cube, rim_width):
    ''' Return IRIS cube with rim removed.
    args
    ----
    cube: input iris cube
    rimwidth: number of grid points to remove from edge of lat and long

    Returns
    -------
    rrcube: rim removed cube
    '''
    # Longitude
    x_coord = cube.coord(axis='X', dim_coords=True)
    # Latitude
    y_coord = cube.coord(axis='Y', dim_coords=True)

    # Remove rim from Longitude
    rrcube = cube.subset(x_coord[rim_width:-1 * rim_width])
    # Remove rim from Latitude
    rrcube = rrcube.subset(y_coord[rim_width:-1 * rim_width])

    return rrcube


if __name__ == '__main__':

    # ===============================================================
    timestamp_start = dt.datetime.now()
    print(f"****** @  {timestamp_start}: Starting processing... **************")

    time_res_mins = 60
    regrid_to_GPM_IMERG=True
    if regrid_to_GPM_IMERG==True:
        grid_length = 10.
    else:
        grid_length = 4.


    #################################################################
    #########============ Read in arguments ===============##########
    #################################################################

    arglist = sys.argv
    print(f"{arglist[1:]}")
    nargs = len(arglist) - 1
    # Command-line arguments provided.
    print(f"Command-line arguments provided: {arglist}")

    test = False

    if test == True:
        #matplotlib.use('Qt5Agg')
        print("TESTING")
        threshold=240
        sm_width_pixels= 5
        halo_pixels= 0
        solve_false_merger= False
        use_halo = True
        dynamic_maxdist= False
        reflectivity_weight_centroids=False
        start_datestr = "20800225"
        end_datestr = "20800401"
        dataset="u-dc139"
        min_area=40000
    else:
        sm_width_pixels = int(arglist[1])
        halo_pixels = int(arglist[2])
        solve_false_merger = arglist[3].lower()=="true"
        dynamic_maxdist = arglist[4].lower()=="true"
        reflectivity_weight_centroids= arglist[5].lower()=="true"  # If using refrlecitivty weighed centroids
        threshold=int(float(arglist[6]))
        start_datestr=str(arglist[7])
        end_datestr=str(arglist[8])
        dataset=str(arglist[9])
        min_area=int(arglist[10])

    print(f"sm_width_pixels: {sm_width_pixels}")
    print(f"halo_pixels: {halo_pixels}")
    print(f"solve_false_merger: {solve_false_merger}")
    print(f"dynamic_maxdist: {dynamic_maxdist}")
    print(f"threshold mm: {threshold}")
    print(f"start_datestr {start_datestr}")
    print(f"end_datestr {end_datestr}")
    print(f"min_area {min_area}")

    startyyyy=str(start_datestr[:4])
    startmm=str(start_datestr[4:6])
    startdd=str(start_datestr[6:8])
    endyyyy=str(end_datestr[:4])
    endmm=str(end_datestr[4:6])
    enddd=str(end_datestr[6:8])

    print("")
    print(f"PROCESSING MCS Tb TRACKS FOR {start_datestr}-{end_datestr}")
    print("")


    # Some 360-day calendars - handle below:
    if dataset in ["u-cw282", "u-cx129", "u-cw288", "u-cx047"]:  # Then these use a 360 day calendar
        start_date = cftime.Datetime360Day(int(startyyyy), int(startmm), int(startdd))  # Start date in 360-day calendar
        end_date = cftime.Datetime360Day(int(endyyyy), int(endmm), int(enddd))  # End date in 360-day calendar
        date_range = generate_360_day_range(start_date, end_date)
        dates2load = [date360.strftime('%Y%m%d') for date360 in date_range]
    elif dataset in ["u-dc272", "u-dc178", "u-dc139", "u-dc077"]:  # Then these use 365 day calendar (no leap years). - timedelta below will correctly handle this
        start_datetime = cftime.DatetimeNoLeap(int(startyyyy), int(startmm), int(startdd), 0, 30) - timedelta(minutes=60)
        end_datetime = cftime.DatetimeNoLeap(int(endyyyy), int(endmm), int(enddd), 23, 30)
        times2load = xr.cftime_range(start=start_datetime, end=end_datetime, freq="D", calendar="noleap").strftime('%Y%m%d')
    else:  # Just use usual Gregorian calendar
        dates2load = pd.date_range(startyyyy+startmm+startdd,endyyyy+endmm+enddd).strftime('%Y%m%d')

    Tbstr = f"{str(threshold)}K"

    halokm = halo_pixels * grid_length  # number of pixels times resolution
    halo = int(halokm / grid_length)  ## Number of pixels
    halosq = halo ** 2  # WK halosq only thing that's used in halo code below
    pad_label = f"halo_{halo_pixels}pixels"


    if dynamic_maxdist == True:  # If using model winds, save to different directory for comparison.
        dynamic_label = 'dynamic_st'
    else:
        dynamic_label = 'fixed'

    if reflectivity_weight_centroids:
        centroid_label = "rwc"  # reflectivity-weighted centroids
    else:
        centroid_label = "geom"  # reflectivity-weighted centroids

    if solve_false_merger:
        false_merger_label = "removed_false_mergers"
    else:
        false_merger_label = "incl_false_mergers"

    sys.stdout.flush()
    min_track_length = None
    pickle_tracks = True


    nt = 1 # Iterated over so unique track PER DAY IF
    hh=0 # hour of track processing. If 0, ALL TRACKS MUST START AS NEWLY INITIATED (.was=-999(

    tracks = []
    for d, date2load in enumerate(dates2load):

        tic()
        # Two different 23:45...
        print(f"Processing {date2load} ({d+1}/{len(dates2load)})")

        ######################### ==== GET RANGE OF TIMES REQUIRED ======#####################
        year = int(date2load[0:4])
        month = int(date2load[4:6])
        day = int(date2load[6:8])

        # curr_date2load below is used to ensure 23:30 on the current (not prev) day is when track info passed over
        if dataset in ["u-cw282", "u-cx129", "u-cw288", "u-cx047"]:  # Then these use a 360 day calendar
            curr_date2load = cftime.Datetime360Day(int(year), int(month), int(day))
        else:
            curr_date2load = datetime(int(year), int(month), int(day))

        if date2load == dates2load[0]: # This means that no previous storms can existst at 23:30 the day before. Fix 00:00 and 23:45
            print("date2load is hard_start_date")
            start_datetime = datetime(int(year), int(month), int(day),0,30)    # Get the first day, 00:30 first
            times2load = pd.date_range(date2load+"0030",date2load+"2330",freq="h").strftime('%Y%m%d_%H%M')
        else: # File will also exist at 23:30, which we want to check against prev day 23:45 to pass over data
            if dataset in ["u-cw282", "u-cx129", "u-cw288", "u-cx047"]:  # Then these use a 360 day calendar
                start_datetime = cftime.Datetime360Day(int(year), int(month), int(day), 0, 30)- timedelta(minutes=60)
                end_datetime =  cftime.Datetime360Day(int(year), int(month), int(day),23,30) # end time always 23:45
                cftimes2load = generate_360_hour_range(start_datetime,end_datetime)
                times2load = [time2load.strftime('%Y%m%d_%H%M') for time2load in cftimes2load]
            elif dataset in ["u-dc272", "u-dc178","u-dc139","u-dc077"]:  # Then these use 365 day calendar (no leap years).
                start_datetime = cftime.DatetimeNoLeap(int(year), int(month), int(day), 0, 30)- timedelta(minutes=60)
                end_datetime = cftime.DatetimeNoLeap(int(year), int(month), int(day), 23, 30)
                times2load = xr.cftime_range(start=start_datetime, end=end_datetime, freq="h", calendar="noleap").strftime('%Y%m%d_%H%M')
            else:
                start_datetime= datetime(int(year), int(month), int(day), 0, 30)- timedelta(minutes=60)  # Get the first day specified.
                end_datetime = datetime(int(year), int(month), int(day),23,30) # end time always 23:45
                times2load = pd.date_range(start_datetime,end_datetime,freq="h").strftime('%Y%m%d_%H%M')
            prev_date2load = start_datetime.strftime('%Y%m%d')  # get prev date
        print(f"{times2load}")

        #####################################################################################

        track_dir = f"/data/users/william.keat/MCS_tracking_fd/DYMECS/{dataset}/Tb/{min_area}km2/tracks/{Tbstr}/{dates2load[0]}-{dates2load[-1]}/{dynamic_label}/{centroid_label}_cen/{time_res_mins}min/sm_{sm_width_pixels}pixels/{pad_label}/{false_merger_label}/on_GPM_{regrid_to_GPM_IMERG}"

        for time2load in times2load: # Loop over all 30 minute pickle times for the selected date above
            print(f"Loading {time2load}")
            sys.stdout.flush()
            year = int(time2load[0:4])
            month = int(time2load[4:6])
            day = int(time2load[6:8])
            hour = int(time2load[9:11])
            minute = int(time2load[11:])

            if dataset in ["u-cw282", "u-cx129", "u-cw288", "u-cx047"]:  # Then these use a 360 day calendar
                curr_time = cftime.Datetime360Day(year, month, day, hour, minute) # datetime of current time
            elif dataset in ["u-dc272", "u-dc178", "u-dc139", "u-dc077"]:  # Then these use 365 day calendar (no leap years).
                curr_time = cftime.DatetimeNoLeap(year, month, day, hour, minute) # datetime of current time
            else:
                curr_time = datetime(year, month, day, hour, minute) # datetime of current time
            curr_hour = curr_time.hour
            prev_time = curr_time - timedelta(minutes=time_res_mins)  # previous time. This won't do anything for first time as all storms unassigned on first stimestep

            file2load = f"/data/scratch/william.keat/MCS_tracking_fd/DYMECS/output/{dataset}/Tb/{Tbstr}/{min_area}km2/{date2load}/{dynamic_label}/{centroid_label}_cen/{time_res_mins}min/sm_{sm_width_pixels}pixels/{pad_label}/{false_merger_label}/on_GPM_{regrid_to_GPM_IMERG}/{time2load}_UTC_storms.p"
            print(f"{file2load}")
            #=================== Load in the storms that from the current day (23:30 unless hard_start date which will be 00:30) ===================#
            with open(file2load, "rb") as curr_pickle:  # Load in storm data for this date and time
                storms = pickle.load(curr_pickle)
            if (date2load == dates2load[0]) & (time2load==times2load[0]): # Then this the first timestep.
                print("First timestep, not stitching, force all storms to initiate")
                # If first date in dates, then we will just assign all storms as "I" and build tracks as usual
                for storm in storms:
                    storm.time = curr_time
                    storm.was = MISVAL # make missing value so that don't have to change code later on.... This forces new track creation
            elif (curr_hour==23): # Then either skip or re-assign
                if (curr_time.day==curr_date2load.day): # This is the first pass... want to skip
                    continue
                else: # This is 23:30 from the following day. Pass over all of these storm properties from 23:30 at the end of the previous day to stitch properly
                    prev_file2load = f"/data/scratch/william.keat/MCS_tracking_fd/DYMECS/output/{dataset}/Tb/{Tbstr}/{min_area}km2/{prev_date2load}/{dynamic_label}/{centroid_label}_cen/{time_res_mins}min/sm_{sm_width_pixels}pixels/{pad_label}/{false_merger_label}/on_GPM_{regrid_to_GPM_IMERG}/{time2load}_UTC_storms.p"
                    print(f"prev_file2load={prev_file2load}")
                    #=================== Load in the storms at overlap time the previous day (23:30) =====================#
                    with open(prev_file2load, "rb") as prev_pickle:  # Load in storm data for this date and time
                        prev_storms = pickle.load(prev_pickle)

                    # u and v components are defined using t and t-1, and assigned to t-1... Therefore end of day 23:30 storm pickles are always -999
                    # However, storms at 23:30 start of next day will be correct. Must be re-assigned
                    storms_u = np.array([storms[i].u for i in np.arange(len(storms))])
                    storms_v = np.array([storms[i].v for i in np.arange(len(storms))])

                    # ALL THAT IS REQUIRED TO STITCH TRACKS:
                    storms=prev_storms.copy() # pass on all the prev storm information to the current storms. No need to iterate (by definition .was at 00:30 will be same storm ids as prev 23:30 by definition due to how labelled.
                    for i in np.arange(len(storms)):
                        storms[i].u = storms_u[i]
                        storms[i].v = storms_v[i]

            # If an overlap time, then set all storms to "U" to be assigned
            for storm in storms:
                storm.time = curr_time
                # Start with "unassigned" status for storm
                storm.status = "U"

            # First create new tracks for new storms
            for storm in storms:
                # This will be different if track was overwritten, where it will be .was of the day before 23:45 storm
                # Implications?
                if storm.was == MISVAL:
                    # Start of a new track
                    storm.status = "I"
                    track = Track(nt, storm)
                    track.datadir = track_dir
                    nt += 1 # This will depend on whether a track existed they day before. Else need to +1 again until unique
                    tracks.append(track)

            # Now remove new storms as no need to process them as there will be no track to build since they are the
            # starting point of a new track by definition
            storms = [storm for storm in storms if storm.status != "I"]

            accreted_storms = []
            child_storms = []
            active_tracks = [track for track in tracks if track.active]

            # Track is active at 00:30, but two storms are at 23:30

            for track in active_tracks:
                # Nothing to do if this track has no storm at the
                # previous time to try and match to
                prev_storm = track.get_storm(prev_time)
                #print(prev_storm)
                if prev_storm is None:
                    continue

                # Now try and find a match to a storm at this time
                match_found = False
                for storm in storms:
                    if storm.was == prev_storm.storm: # Potential issue here if .was different (after track override)
                        match_found = True
                        if storm.parent == [MISVAL] and storm.child == [MISVAL] and storm.accreted == [MISVAL]:
                            # Straightforward continuation of track (no split or merge)
                            storm.status = "C"
                            track.add_storm(storm) # here is where "problem" storm is being added with duplicate time

                        if storm.parent == [MISVAL] and storm.child == [MISVAL] and storm.accreted != [MISVAL]:
                            # This storm has accreted other storms (merger) and is thus
                            # the continuation of the track
                            # The tracks of the accreted storms will be terminated below
                            storm.status = "MC"
                            track.add_storm(storm)
                            accreted_storms.extend(storm.accreted)

                        if storm.parent == [MISVAL] and storm.child != [MISVAL] and storm.accreted == [MISVAL]:
                            # This storm is a child of another storm (split)
                            # We will start a new track for this storm below
                            storm.status = "SI"
                            storm.was = MISVAL
                            child_storms.append(storm)

                        if storm.parent == [MISVAL] and storm.child != [MISVAL] and storm.accreted != [MISVAL]:
                            # This storm is a child of another storm (split) and has accreted storms (merge)
                            # We will start a new track for this storm and terminate the accreted storm
                            # tracks below
                            storm.status = "SI,MC"
                            storm.was = MISVAL
                            accreted_storms.extend(storm.accreted)
                            child_storms.append(storm)

                        if storm.parent != [MISVAL] and storm.child == [MISVAL] and storm.accreted == [MISVAL]:
                            # This storm is a parent of child storms (split) and is thus
                            # the continuation of the track
                            storm.status = "SC"
                            track.add_storm(storm)

                        if storm.parent != [MISVAL] and storm.child == [MISVAL] and storm.accreted != [MISVAL]:
                            # This storm is a parent of child storms (split) and has accreted storms (merge)
                            # We will terminate the accreted storm tracks below
                            storm.status = "SC,MC"
                            track.add_storm(storm)
                            accreted_storms.extend(storm.accreted)

                        if storm.parent != [MISVAL] and storm.child != [MISVAL] and storm.accreted == [MISVAL]:
                            # Don't think this one should ever happen
                            raise ValueError("Storm simultaneously a parent of a storm and a child of a storm")
                        #                                 storm.status = "SC,SI"
                        #                                 storm.was = MISVAL
                        #                                 child_storms.append(storm)

                        if storm.parent != [MISVAL] and storm.child != [MISVAL] and storm.accreted != [MISVAL]:
                            # Don't think this one should ever happen
                            raise ValueError(
                                "Storm simultaneously a parent of a storm and a child of a storm and has accreted storms")
                #                                 storm.status = "SC,SI,MC"
                #                                 storm.was = MISVAL
                #                                 accreted_storms.extend(storm.accreted)
                #                                 child_storms.append(storm)

                if not match_found:
                    # This track terminates
                    prev_storm.status = "T"
                    track.active = False



            print(f"{time2load} NUMBER TRACKS = {len(tracks)}")
            sys.stdout.flush()

            # Deal with splits - make a new track for
            # any child storms that split from another storm
            for storm in child_storms:
                track = Track(nt, storm)
                track.datadir = track_dir
                nt += 1
                tracks.append(track)

            # Deal with track termination by merger
            # print(accreted_storms)
            for track in tracks:
                prev_storm = track.get_storm(prev_time)
                if prev_storm is None:
                    continue
                if prev_storm.storm in accreted_storms:
                    # Flag that this storm is accreted by another at
                    # the next timestep and thus its track terminates
                    prev_storm.status = "MT"
                    track.active = False

            # Make sure there are no unassigned storms left at this point
            unassigned_storms = [storm for storm in storms if storm.status == "U"]
            list_of_unassigned_storms = []
            if len(unassigned_storms) > 0:
                raise ValueError("Shouldn't be any unassigned storms at this point")
            sys.stdout.flush()

            hh=hh+1
        toc()

    if pickle_tracks:
        # Save tracks to directory (range of dates if in chunks)
        if not os.path.exists(track_dir):
            os.makedirs(track_dir)
            print(f"Created dir : {track_dir} ")
        pickle_file = f"{track_dir}/tracks.p"
        print(f"Writing tracks to pickle file {pickle_file}")
        with open(pickle_file, "wb") as open_pickle:
            pickle.dump(tracks, open_pickle)
            print(f"Tracks saved here:{pickle_file}")

    tot_nstorms = 0
    for track in tracks:
        tot_nstorms += len(track.storms)
    print(f"Total number of storms: {tot_nstorms}")
    print(f"Total number of tracks: {len(tracks)}")

    # Throw away short tracks
    if min_track_length is not None:
        tracks = [track for track in tracks if track.get_lifetime() > min_track_length]
        print(f"Total number of tracks of length > {min_track_length}: {len(tracks)}")
    sys.stdout.flush()

    # Do some checking
    active_tracks = [track for track in tracks if track.active]
    print(f"Total number of active tracks: {len(active_tracks)}")
    inactive_tracks = [track for track in tracks if not track.active]
    print(f"Total number of inactive tracks: {len(inactive_tracks)}")

    tot_nstorms = 0
    for track in inactive_tracks:
        tot_nstorms += len(track.storms)
    print(f"Total number of storms: {tot_nstorms}")

    n_simple = 0
    n_init = 0
    n_split_init = 0
    n_single = 0
    n_accreted = 0
    n_at_least_one_merger = 0
    n_at_least_one_split = 0
    n_at_least_one_merger_or_one_split = 0
    n_primary_tracked = 0
    n_secondary_tracked = 0
    for track in inactive_tracks:
        statuses = track.get_statuses()
        unique_statuses = set(statuses)

        primary_tracked = track.is_primary_tracked()
        n_primary_tracked += primary_tracked.count(True)
        n_secondary_tracked += primary_tracked.count(False)

        if statuses.count("I") > 1:
            raise ValueError("I status can only appear once")

        if statuses.count("SI") > 1:
            raise ValueError("SI status can only appear once")

        if track.active:
            if statuses.count("T") != 0:
                raise ValueError("Active track can't have T status anywhere")
            if statuses.count("MT") != 0:
                raise ValueError("Active track can't have MT status anywhere")
        else:
            if statuses.count("T") > 1:
                raise ValueError("Dead track can only have T status once")
            if statuses.count("MT") > 1:
                raise ValueError("Dead track can only have MT status once")

        if not track.active and statuses[-1] not in ["T", "MT"]:
            raise ValueError("The last storm status in an inactive track must be T or MT")

        if set(["I", "SI"]) == unique_statuses:
            raise ValueError("Cannot have both I and SI statuses in a track")

        # Count the number of tracks which are one timestep long
        if len(statuses) == 1:
            n_single += 1

        # Count the number of simple tracks (I -> C or I -> C -> T)
        if track.active:
            if set(["I", "C"]) == unique_statuses:
                n_simple += 1
        else:
            if set(["I", "T"]) == unique_statuses or set(["I", "C", "T"]) == unique_statuses:
                n_simple += 1

        # Count the number of tracks truly initiated (i.e. not by splitting off a parent cell)
        if "I" in unique_statuses:
            n_init += 1

        # Count the number of tracks initiated following a split
        if "SI" in unique_statuses:
            n_split_init += 1

        # Count the number of tracks that were terminated by merging into another cell
        if "MT" in unique_statuses:
            n_accreted += 1

        # Count the number of tracks which accreted other cells at least once in their lifetime
        if "MC" in unique_statuses:
            n_at_least_one_merger += 1

        # Count the number of tracks which split off child cells at least once in their lifetime
        if "SC" in unique_statuses:
            n_at_least_one_split += 1

        # Count the number of tracks which accreted other cells at least once in their lifetime OR split off child cells at least once in their lifetime
        if "MC" in unique_statuses or "SC" in unique_statuses:
            n_at_least_one_merger_or_one_split += 1

    print(f"Total number of tracks of length 1: {n_single}")
    print(f"Total number of simple tracks: {n_simple}")
    print(f"Total number of tracks truly initiated: {n_init}")
    print(f"Total number of tracks initiated by splitting: {n_split_init}")
    print(f"Total number of tracks terminated by merging: {n_accreted}")
    print(f"Total number of tracks with at least one merger event: {n_at_least_one_merger}")
    print(f"Total number of tracks with at least one split event: {n_at_least_one_split}")
    print("Total number of tracks with at least one merger or at least one split event:",
          n_at_least_one_merger_or_one_split)
    print(f"Total number of primary tracked storms: {n_primary_tracked}")
    print(f"Total number of secondary tracked storms: {n_secondary_tracked}")
    print()
    sys.stdout.flush()

    statuses = []
    # for track in track_dict[name]:
    for track in tracks:
        for storm in track.storms:
            statuses.append(storm.status)
    for status in ["I", "C", "MC", "SI", "SI,MC", "SC", "SC,MC", "SC,SI", "SC,SI,MC", "MT", "T", "U"]:
        print(f"{status} {statuses.count(status)}")
    print(f"Total {len(statuses)}")
    sys.stdout.flush()
    timestamp_end = dt.datetime.now()
    print(f"****** @  {timestamp_end}: Finished! **************")
    timediff_total = timestamp_end - timestamp_start
    print(f"########## Total run time:  {timediff_total}\n")

    print("===========================================================")
    print("           ALL DONE!")
    print("===========================================================")