'''
Created on 11/04/2022

@author: wkeat
Written to allow multiple configurations/dates of cell tracking code to be run in parallel
'''

'''
This launches MCS_tracking.py
'''
import numpy as np
import sys
import pandas as pd
import time
import cftime
import track_period_dicts as tpd
import xarray as xr

class para_args(object):

    '''
    Defines a class to collate the calling arguments to functions run in parallel with the
    pool method.
    Usage: Loop as in serial processing to generate a list of all calling arguments.
           outside the loop, open a parallel pool region and map the list of calling
           arguments
    '''

    def __init__(self, cmd=None, msg=None):
        self.cmd = cmd
        self.msg = msg


def shellcmd(args):
    import subprocess
    try:
        retcode = subprocess.call(args.cmd, shell=True)
        if retcode < 0:
            print('syst.cmd terminated by signal', retcode)
        elif retcode:
            print('syst.cmd returned in ', args.msg, '', retcode)
    except OSError as ex:
        print("Execution failed in " + args.msg + ": ", ex)


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



if __name__ == "__main__":
    # INPUT
    # ===========================================================================================#
    # ============================= OBSERVATIONS/ERA-DRIVEN SUITES   ============================#
    # ===========================================================================================#
    datasets=["GPM"]#,"u-df283"] # GPM for observations, suite-id for all else.
    sm_width_pixel_opts = np.array([5])  # ,1,2,3,4,5]) # range of smoothing # pixels to test
    halo_pixel_opts = np.array([0])  # 0,1,2,3,4,5]) # pixel radius of halo or padding applied for secondary tracking
    solve_false_merger_opts = [False]  # True = erosion/dilation step
    reflectivity_weight_centroids_opts = [False]
    dynamic_maxdist_opts = ["False"]
    min_areas=[40000]
    thresholds = [240]  # K for cold cloud shield

    # ============================================================================================#
    # SPECIFY SPICE SETTINGS
    ntasks = 1
    mem_req = '18G'  #
    queue = "normal"
    time_req = 3 #minutes


    for dataset in datasets:
        if 0:  # Full range of the dataset
            date_start = tpd.suite_date_ranges[dataset]["start_date"] # I had date ranges stored here in dictionary
            date_end = tpd.suite_date_ranges[dataset]["end_date"]
        else:  # Use custom range rather than dictionary
            date_start = "19981201"  # Start date in YYYYMMDD format
            date_end = "19990101"  # End date in YYYYMMDD format

        startyyyy=str(date_start[:4])
        startmm=str(date_start[4:6])
        startdd=str(date_start[6:8])
        endyyyy=str(date_end[:4])
        endmm=str(date_end[4:6])
        enddd=str(date_end[6:8])

        # Some 360/365-day calendars - handle below:
        if dataset in ["u-cw282", "u-cx129", "u-cw288", "u-cx047"]:  # Then these use a 360 day calendar. Define date_range using cftime
            # Step 1: Define start and end dates
            if enddd!= 30:
                enddd = 30  # if going to the end of a month, set always to 30 if not.
                date_end = str(endyyyy) + str(endmm) + str(enddd)  # update end date
            start_date = cftime.Datetime360Day(int(startyyyy), int(startmm), int(startdd))  # Start date in 360-day calendar
            end_date = cftime.Datetime360Day(int(endyyyy), int(endmm), int(enddd))  # End date in 360-day calendar
            date_range = generate_360_day_range(start_date, end_date)
            date_range = [date360.strftime('%Y%m%d') for date360 in date_range]

        elif dataset in ["u-dc272","u-dc178","u-dc077","u-dc139"]: # Then these use 365 day calendar (no leap years). Define date_range using pandas but excluding 29th Feb in leap years
            start_date = cftime.DatetimeNoLeap(int(startyyyy), int(startmm), int(startdd))  # Start date in 360-day calendar
            end_date = cftime.DatetimeNoLeap(int(endyyyy), int(endmm), int(enddd))  # End date in 360-day calendar
            date_range=xr.cftime_range(start=start_date, end=end_date, freq="D", calendar="noleap").strftime('%Y%m%d')

        else:  # Just use usual Gregorian calendar
            date_range= pd.date_range(startyyyy+startmm+startdd,endyyyy+endmm+enddd).strftime('%Y%m%d')


        # SUBMIT A LOAD OF JOBS TO SPICE FOR EACH DATE AND EACH COMBINATION OF SETTINGS
        for datestr in date_range:  # iterate over each break date
            time.sleep(0.5) # quick sleep - this means that won't overload SPICE as jobs only take a about 1-2 mins and will be complete before last submitted
            for sm_width in sm_width_pixel_opts:
                for halo_pixels in halo_pixel_opts:
                    for solve_false_merger_opt in solve_false_merger_opts:
                        for dynamic_maxdist in dynamic_maxdist_opts:
                            for rwc in reflectivity_weight_centroids_opts:
                                for threshold in thresholds:
                                    for min_area in min_areas:
                                        time.sleep(0.2)
                                        scienv = 'module load scitools'
                                        cmd = f'python /home/users/william.keat/workspace/MCSs/MCS_OLR_tracking_full_domain.py {sm_width} {halo_pixels} {solve_false_merger_opt} {dynamic_maxdist} {rwc} {datestr} {threshold} {dataset} {min_area}'
                                        print(cmd)
                                        wrapper_spice = f'/home/users/william.keat/scripts/MCS_OLR_tracking_full_domain_on_spice_{sm_width}_{halo_pixels}_{solve_false_merger_opt}_{dynamic_maxdist}_{rwc}_{datestr}_{threshold}_{dataset}_{min_area}.batch'
                                        with open(wrapper_spice, "w") as fh:
                                            fh.write('#!/bin/bash -l\n')
                                            fh.write(f'#SBATCH --mem={mem_req}\n')
                                            fh.write(f'#SBATCH --ntasks={ntasks}\n')
                                            fh.write(f'#SBATCH --output=/home/users/william.keat/python/spice_out/MCS_{dataset}_OLR_fd_cell_tracking_{sm_width}_{halo_pixels}_{solve_false_merger_opt}_{dynamic_maxdist}_{rwc}_{datestr}_{threshold}_{min_area}.out\n')
                                            fh.write(f'#SBATCH --error=/home/users/william.keat/python/spice_out/MCS_{dataset}_OLR_fd_cell_tracking_{sm_width}_{halo_pixels}_{solve_false_merger_opt}_{dynamic_maxdist}_{rwc}_{datestr}_{threshold}_{min_area}.err\n')
                                            fh.write(f'#SBATCH --time={time_req}\n')
                                            fh.write(f'#SBATCH --qos={queue}\n')

                                            fh.write('\n')
                                            fh.write(f'{scienv}\n')
                                            fh.write('\n')
                                            fh.write(f'{cmd}\n')
                                            fh.write('\n')

                                        print(f'launching {cmd}')
                                        argument = para_args(
                                            cmd=f'sbatch /home/users/william.keat/scripts/MCS_OLR_tracking_full_domain_on_spice_{sm_width}_{halo_pixels}_{solve_false_merger_opt}_{dynamic_maxdist}_{rwc}_{datestr}_{threshold}_{dataset}_{min_area}.batch',
                                            msg='failed to submit to SPICE')
                                        shellcmd(argument)