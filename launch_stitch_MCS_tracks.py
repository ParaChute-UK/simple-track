'''
Created on 01/03/2024

@author: wkeat
This launches stitch_MCS_tracks_fd.py
'''

import numpy as np
import sys
import pandas as pd
import time
import track_period_dicts as tpd

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


#########################################################
# INPUT
sm_width_pixel_opts = np.array([5])  # ,1,2,3,4,5]) # range of smoothing # pixels to test
halo_pixel_opts = np.array([0])  # 0,1,2,3,4,5]) # pixel radius of halo or padding applied for secondary tracking
solve_false_merger_opts = [False]  # True = erosion/dilation step
reflectivity_weight_centroids_opts = [False]
dynamic_maxdist_opts = ["False"]
min_areas=[40000]
thresholds = [240]  # K for cold cloud shield
datasets=["GPM","u-df283"]


ntasks = 1
mem_req = '200G' #
queue = "long"
time_req = 720 # Quick at first but scales horribly as track number increases

partition="cpu-long"
# if min_area>10000: # MCS, need less time
#     ntasks = 1
#     mem_req = '75G'  # Was 70G
#     queue = "normal"
#     time_req = 360
# else:
#     ntasks = 1
#     mem_req = '18G'  # Was 70G
#     queue = "normal"
#     time_req = 120


for dataset in datasets:
    if 1:  # Full range
        date_start = tpd.suite_date_ranges[dataset]["start_date"]
        date_end = tpd.suite_date_ranges[dataset]["end_date"]
    else: # custom range
        date_start = "20000101"  # Start date in YYYYMMDD format
        date_end = "20071230"  # End date in YYYYMMDD format

    for sm_width in sm_width_pixel_opts:
        for halo_pixels in halo_pixel_opts:
            for solve_false_merger_opt in solve_false_merger_opts:
                for dynamic_maxdist in dynamic_maxdist_opts:
                    for rwc in reflectivity_weight_centroids_opts:
                        for threshold in thresholds:
                            for min_area in min_areas:
                                time.sleep(0.2)
                                scienv = 'module load scitools'
                                cmd = f'python /home/users/william.keat/workspace/MCSs/process_MCS_OLR_tracks_fd.py {sm_width} {halo_pixels} {solve_false_merger_opt} {dynamic_maxdist} {rwc} {threshold} {date_start} {date_end} {dataset} {min_area}'
                                print(cmd)
                                wrapper_spice = f'/home/users/william.keat/scripts/process_MCS_OLR_tracks_fd_on_spice_{sm_width}_{halo_pixels}_{solve_false_merger_opt}_{dynamic_maxdist}_{rwc}_{threshold}_{date_start}_{date_end}_{dataset}_{min_area}.batch'
                                with open(wrapper_spice, "w") as fh:
                                    fh.write('#!/bin/bash -l\n')
                                    fh.write(f'#SBATCH --mem={mem_req}\n')
                                    fh.write(f'#SBATCH --ntasks={ntasks}\n')
                                    fh.write(f'#SBATCH --output=/home/users/william.keat/python/spice_out/process_MCS_OLR_tracks_fd_{sm_width}_{halo_pixels}_{solve_false_merger_opt}_{dynamic_maxdist}_{rwc}_{threshold}_{date_start}_{date_end}_{dataset}_{min_area}.out\n')
                                    fh.write(f'#SBATCH --error=/home/users/william.keat/python/spice_out/process_MCS_OLR_tracks_fd_{sm_width}_{halo_pixels}_{solve_false_merger_opt}_{dynamic_maxdist}_{rwc}_{threshold}_{date_start}_{date_end}_{dataset}_{min_area}.err\n')
                                    fh.write(f'#SBATCH --time={time_req}\n')
                                    if queue=="long":
                                        fh.write('#SBATCH --partition=cpu-long\n')
                                    else:
                                        fh.write(f'#SBATCH --qos={queue}\n')

                                    fh.write('\n')
                                    fh.write(f'{scienv}\n')
                                    fh.write('\n')
                                    fh.write(f'{cmd}\n')
                                    fh.write('\n')

                                print(f'launching {cmd}')
                                argument = para_args(
                                    cmd=f'sbatch /home/users/william.keat/scripts/process_MCS_OLR_tracks_fd_on_spice_{sm_width}_{halo_pixels}_{solve_false_merger_opt}_{dynamic_maxdist}_{rwc}_{threshold}_{date_start}_{date_end}_{dataset}_{min_area}.batch',
                                    msg='failed to submit to SPICE')
                                shellcmd(argument)