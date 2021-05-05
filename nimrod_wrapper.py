import datetime
from glob import glob
import sys

import numpy as np

import object_tracking
import nimrod_object_tracking
import nimrod_user_functions

if sys.argv[1] == 'orig':
    ot = object_tracking
elif sys.argv[1] == 'nimrod':
    ot = nimrod_object_tracking

##################################################################
# THE FOLLOWING PARAMETERS SHOULD BE CHANGED BASED ON THE DATA (RESOLUTION ETC.)
##################################################################

# Integer number (dimensions user-defined) to identify MINIMUM time difference between consecutive data files
# Example 1: Radar data 5-minutes apart with time stamp in filename, dt = 5
# Example 2: Satellite brightness temperatures hourly with time stamp in filename, dt = 1
# NB. When writing storms, (dx,dy) will have units PIXELS per TIME STEP (specified by dt), so already scaled by number of missing files

dt = 5.
dt_tolerance = 15.  # Maximum separation in time allowed between consecutive images

under_t = False  ## True = labelling areas *under* the threshold (e.g. brightness temperature), False = labelling areas *above* threshold (e.g. rainfall)
threshold = 3.  ## Threshold used to identify objects (with value of variable greater than this threshold)
minpixel = 4.  ## Minimum object size in pixels
squarelength = 200.  ## Size in pixels of individual squares to run fft for (dx,dy) displacement. Must divide (x,y) lengths of the array!
rafraction = 0.01  ## Minimum fractional cover of objects required for fft to obtain (dx,dy) displacement
dd_tolerance = 3.  # Maximum difference in displacement values between adjacent squares (to remove spurious values) - scaled by num_dt if necessary
halopixel = 5.  ## Radius of halo in pixels for orphan storms - big halo assumes storms may spawn "children" at a distance multiple pixels away

flagwrite = True  ## For writing storm history data in a text file
doradar = False  ## doradar=True is calculate range and azimuth for real-time tracking with radar (e.g. Chilbolton). doradar=False any other use, radar coordinates not relevant
misval = -999  ## Missing value
struct2d = np.ones(
    (3, 3))  ## np.ones((3,3)) is 8-point connectivity for labelling storms. Can be changed to user preference.

flagplot = False  ## For plotting images (vectors and IDs). Also set plot_type...
flagplottest = False  ## For plotting fft correlations (testing only, very slow, lots of plots)

if flagplot or flagplottest:
    plot_type = '.png'
    if plot_type == '.eps':
        my_dpi_global = 150  # for rasterized plots, this controls the size and quality of the final plot
    elif plot_type == '.png':
        my_dpi_global = 300

##################################################################
# THE FOLLOWING PARAMETERS CAN BE CHANGED, BUT SHOULD NOT BE
##################################################################

lapthresh = 0.6  ## Minimum fraction of overlap (0.6 in TITAN)

##################################################################
# AUTOMATIC SET UP OF DISCRETE VALUES BASED ON USER INPUT PARAMETERS
# ESSENTIAL - NO NEED TO CHANGE THESE
##################################################################
# squarehalf: To determine grid spacing for coarse grid of (dx,dy) estimates
# areastr: For filename identifier of area threshold used
# thr_str: For filename identifier of variable threshold used
# fftpixels: Minimum number of thresholded pixels needed to calculate (dx,dy)
# halosq: To identify if new cell is nearby existing cell
##################################################################

squarehalf = int(squarelength / 2)
areastr = str(int(minpixel))
thr_str = str(int(threshold))
sql_str = str(int(squarelength))
fftpixels = squarelength**2/int(1./rafraction)
# fftpixels = 30
halosq = halopixel ** 2

##################################################################
# AUTOMATIC SET UP OF TEXT STRING FOR INFORMATION ON LABELLING
# NOT ESSENTIAL
##################################################################

label_method = 'Rainfall rate > ' + thr_str + 'mm/hr'

##################################################################
# THE REMAINDER IS THE SET UP FOR THE EXAMPLE DATA
# THIS SHOULD BE ADJUSTED RELEVANT TO THE USER DATA
# !!!TEST: MAKE SURE xall AND yall DIVIDE IN squarelength!!!
##################################################################

# xmat, ymat = np.meshgrid(range(-200, 200), range(-150, 150))
xmat, ymat = np.meshgrid(range(-400, 400), range(-300, 300))
xall = np.size(xmat, 0)  # Only used to check grid dimensions
yall = np.size(xmat, 1)  # Only used to check grid dimensions
if (np.fmod(xall, squarelength) != 0 or np.fmod(yall, squarelength) != 0):
    raise ValueError('Your grid does not match a multiple of squares as defined by squarelength')

#################################################################
# For test data, try the following:
#
# All data (5-minute intervals)
# DATA_DIR = './data/'
# IMAGES_DIR = './output/'
# Sparse data (10-minute intervals), to test similarity in vector fields (scaling by num_dt working correctly)
# DATA_DIR = './data/'
# IMAGES_DIR = './output/'
# Missing data (10-minute intervals, 1 file missing), to test dt_tolerance
# DATA_DIR = './data/'
# IMAGES_DIR = './output/'
#################################################################
DATA_DIR = '/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data/wescon/output/2012/06/'
IMAGES_DIR = './output/'
filelist = sorted(glob(DATA_DIR + 'metoffice-c-band-rain-radar_uk_201206*.nc'))

#   Initialise variables
OldData, OldLabels, oldvar, newvar, prev_time = [], [], [], [], []
newwas = 1
plot_vectors = False

start_time = None
oldhourval = []
oldminval = []
oldmask = []
newmask = []
num_dt = []

chilbolton_centred = True
loader = nimrod_user_functions.FileLoader(filelist, chilbolton_centred=chilbolton_centred)

for nt, (var, file_ID, now_time) in enumerate(loader.load_next()):
    if not start_time:
        start_time = now_time
    if now_time.minute == 0 and now_time.hour == 0:
        flagplot = True
        flagplottest = True
    else:
        flagplot = False
        flagplottest = False

    # Load new image
    # now_time = start_time + datetime.timedelta(seconds=300. * nt)
    # var,file_ID,hourval,minval = nimrod_user_functions.loadfile(DATA_DIR + filelist[nt])
    print(file_ID)
    domain = 'chil_' if chilbolton_centred else 'central_'
    write_file_ID = domain + 'S' + sql_str + '_T' + thr_str + '_A' + areastr + '_' + file_ID
    NewLabels = ot.label_storms(var, minpixel, threshold, struct2d, under_t)
    # oldmask, newmask, USED FOR DERIVING (dx,dy)
    # THESE CAN BE CHANGED USING EXPERT KNOWLEDGE (e.g. use raw data rather than binary masks, if displacement information is contained in structures within objects)
    # !!! NB If raw data are used (i.e. not zeros and ones) then fftpixels needs to be changed to remain sensible !!!
    if len(OldLabels) > 1:
        # CHECK TIME DIFFERENCE BETWEEN CONSECUTIVE IMAGES
        # dtnow = nimrod_user_functions.timediff(oldhourval, oldminval, hourval, minval)
        dtnow = (now_time - old_time).total_seconds() / 60  # timediff in minutes.
        num_dt = dtnow / dt
        if dtnow > dt_tolerance:
            print('Data are too far apart in time --- Re-initialise objects')
            OldData, OldLabels, oldvar, newvar, prev_time = [], [], [], [], []
            newwas = 1
            plot_vectors = False
            continue
        oldmask = np.where(OldLabels >= 1, 1, 0)
        newmask = np.where(NewLabels >= 1, 1, 0)
    # Call object tracking routine
    # NewData = list of objects and properties
    # newwas = final label number
    # NewLabels = array with object IDs from [1, nummax] as found by label_storms
    # newumat, newvmat = arrays with (dx,dy) displacement between two images (NB not displacement per dt!!!)
    # wasarray = array with object IDs consistent across images (i.e. tracked IDs)
    # lifearray = array with object lifetime consistent across images
    (NewData, newwas, NewLabels,
     newumat, newvmat, wasarray, lifearray) = ot.track_storms(OldData, var, newwas, NewLabels,
                                                              OldLabels, xmat, ymat, fftpixels,
                                                              dd_tolerance, halosq, squarehalf,
                                                              oldmask, newmask, num_dt, lapthresh,
                                                              misval, doradar, under_t, IMAGES_DIR,
                                                              write_file_ID, flagplottest)
    # Write tracked storm information (see ot.write_storms)
    if flagwrite:
        ot.write_storms(write_file_ID, start_time, now_time, label_method, squarelength, rafraction,
                        newwas, NewData, doradar, misval, IMAGES_DIR)
    # Plot tracked storm information (see nimrod_user_functions.plot_example)
    if flagplot:
        try:
            nimrod_user_functions.plot_example(write_file_ID, nt, var, xmat, ymat, newumat, newvmat, num_dt, wasarray,
                                               lifearray, threshold, IMAGES_DIR, plot_vectors)
        except:
            pass
    # Save tracking information in preparation for next image
    OldData = NewData
    OldLabels = NewLabels
    oldvar = var
    old_time = now_time
    plot_vectors = True
