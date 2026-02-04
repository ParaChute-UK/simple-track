import object_tracking
import numpy as np
import datetime
import os
import user_functions
import cv2

##################################################################
# THE FOLLOWING PARAMETERS SHOULD BE CHANGED BASED ON THE DATA (RESOLUTION ETC.)
##################################################################

# Integer number (dimensions user-defined) to identify MINIMUM time difference between consecutive data files
# Example 1: Radar data 5-minutes apart with time stamp in filename, dt = 5
# Example 2: Satellite brightness temperatures hourly with time stamp in filename, dt = 1
# NB. When writing storms, (dx,dy) will have units PIXELS per TIME STEP (specified by dt), so already scaled by number of missing files
dt = 5. 
dt_tolerance = 15. # Maximum separation in time allowed between consecutive images

under_t=False ## True = labelling areas *under* the threshold (e.g. brightness temperature), False = labelling areas *above* threshold (e.g. rainfall)
threshold = 3. ## Threshold used to identify objects (with value of variable greater than this threshold)
minpixel = 4. ## Minimum object size in pixels
squarelength = 100. ## Size in pixels of individual squares to run fft for (dx,dy) displacement. Must divide (x,y) lengths of the array!
rafraction = 0.01 ## Minimum fractional cover of objects required for fft to obtain (dx,dy) displacement
dd_tolerance = 3. # Maximum difference in displacement values between adjacent squares (to remove spurious values) - scaled by num_dt if necessary
halopixel = 5. ## Radius of halo in pixels for orphan storms - big halo assumes storms may spawn "children" at a distance multiple pixels away

flagwrite = True ## For writing storm history data in a text file 
doradar = False ## doradar=True is calculate range and azimuth for real-time tracking with radar (e.g. Chilbolton). doradar=False any other use, radar coordinates not relevant
misval = -999 ## Missing value
struct2d = np.ones((3,3)) ## np.ones((3,3)) is 8-point connectivity for labelling storms. Can be changed to user preference.

flagplot = True ## For plotting images (vectors and IDs). Also set plot_type...
flagplottest = False ## For plotting fft correlations (testing only, very slow, lots of plots)

if flagplot or flagplottest:
    plot_type = '.png'
    if plot_type == '.eps':
        my_dpi_global = 150 # for rasterized plots, this controls the size and quality of the final plot
    elif plot_type == '.png':
        my_dpi_global = 300

##################################################################
# THE FOLLOWING PARAMETERS CAN BE CHANGED, BUT SHOULD NOT BE
##################################################################

lapthresh = 0.6 ## Minimum fraction of overlap (0.6 in TITAN)

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

squarehalf = int(squarelength/2)   
areastr = str(int(minpixel))
thr_str = str(int(threshold))
sql_str = str(int(squarelength))
fftpixels = squarelength**2/int(1./rafraction)
halosq = halopixel**2

##################################################################
# AUTOMATIC SET UP OF TEXT STRING FOR INFORMATION ON LABELLING
# NOT ESSENTIAL
##################################################################

label_method='Rainfall rate > ' + thr_str + 'mm/hr'    

##################################################################
# THE REMAINDER IS THE SET UP FOR THE EXAMPLE DATA
# THIS SHOULD BE ADJUSTED RELEVANT TO THE USER DATA
# !!!TEST: MAKE SURE xall AND yall DIVIDE IN squarelength!!!
##################################################################

xmat,ymat = np.meshgrid(range(-200,200),range(-150,150))
xall = np.size(xmat,0) # Only used to check grid dimensions
yall = np.size(xmat,1) # Only used to check grid dimensions
if (np.fmod(xall,squarelength)!=0 or np.fmod(yall,squarelength)!=0):
	raise ValueError('Your grid does not match a multiple of squares as defined by squarelength')

#################################################################
# For test data, try the following:
# 
# All data (5-minute intervals)
#DATA_DIR = './data/'
#IMAGES_DIR = './output/'
# Sparse data (10-minute intervals), to test similarity in vector fields (scaling by num_dt working correctly)
#DATA_DIR = './data/'
#IMAGES_DIR = './output/'
# Missing data (10-minute intervals, 1 file missing), to test dt_tolerance
#DATA_DIR = './data/'
#IMAGES_DIR = './output/'
#################################################################
DATA_DIR = './data/'
IMAGES_DIR = './output/'
filelist = os.listdir(DATA_DIR)
filelist = np.sort(filelist)
if doradar:
	rarray=np.sqrt(xmat**2+ymat**2);
	azarray=np.arctan(xmat/ymat);
	azarray[np.where((xmat<0) & (ymat>=0))]=azarray[np.where((xmat<0) & (ymat>=0))]+2*np.pi;
	azarray[np.where(ymat<0)]=azarray[np.where(ymat<0)]+np.pi; 
	azarray=180*azarray/np.pi;    
	azarray[np.where(np.isnan(azarray)==1)]=0         

#   Initialise variables
OldData, OldLabels, oldvar, newvar, prev_time = [], [], [], [], []
newwas = 1
plot_vectors = False

start_time = datetime.datetime(2012,8,25,14,5,0,0)
oldhourval = []
oldminval = []
oldmask = []
newmask = []
num_dt = []

###### MODIFIED OPTICAL FLOW CODE CONTRIBUTED BY CHRIS SHORT #######
use_new_opt_flow = True
of_method = "DIS"
direction = "backward"
    
# Below functions heavily based on REF
def scaler(data):
    c1 = data.min()
    c2 = data.max()
    return ((data - c1) / (c2 - c1) * 255).astype(np.uint8), c1, c2

def fill_holes(of_instance, threshold=0):
    # calculate velocity scalar
    vlcty = np.sqrt(of_instance[::, ::, 0]**2 + of_instance[::, ::, 1]**2)

    # zero mask
    zero_holes = vlcty <= threshold
    if vlcty.size == np.count_nonzero(zero_holes):
        print("WARNING: vlcty <= 0 everywhere")
        delta_x_target = np.zeros(vlcty.shape)
        delta_y_target = np.zeros(vlcty.shape)
    else:
        # targets
        coord_target_i, coord_target_j = np.meshgrid(range(of_instance.shape[1]),
                                                     range(of_instance.shape[0]))

        # source
        coord_source_i, coord_source_j = coord_target_i[~zero_holes], coord_target_j[~zero_holes]
        delta_x_source = of_instance[::, ::, 0][~zero_holes]
        delta_y_source = of_instance[::, ::, 1][~zero_holes]

        # reshape
        src = np.vstack((coord_source_i.ravel(), coord_source_j.ravel())).T
        trg = np.vstack((coord_target_i.ravel(), coord_target_j.ravel())).T

        # create an object
        import wradlib.ipol as ipol
        interpolator = ipol.Idw(src, trg)

        # do interpolation
        delta_x_target = interpolator(delta_x_source.ravel())
        delta_y_target = interpolator(delta_y_source.ravel())

        # reshape output
        delta_x_target = delta_x_target.reshape(of_instance.shape[0],
                                                of_instance.shape[1])
        delta_y_target = delta_y_target.reshape(of_instance.shape[0],
                                                of_instance.shape[1])

    return np.stack([delta_x_target, delta_y_target], axis=-1)

def calculate_of(data_instance,
                 method="DIS",
                 direction="forward"):
    # define frames order
    if direction == "forward":
        prev_frame = data_instance[-2]
        next_frame = data_instance[-1]
        coef = 1.0
    elif direction == "backward":
        prev_frame = data_instance[-1]
        next_frame = data_instance[-2]
        coef = -1.0

    # calculate dense flow
    if method == "Farneback":
        of_instance = cv2.optflow.createOptFlow_Farneback()
    elif method == "DIS":
        # Name depends on version of CV2
        # of_instance = cv2.optflow.createOptFlow_DIS()
        of_instance = cv2.DISOpticalFlow_create()
    elif method == "DeepFlow":
        of_instance = cv2.optflow.createOptFlow_DeepFlow()
    elif method == "PCAFlow":
        of_instance = cv2.optflow.createOptFlow_PCAFlow()
    elif method == "SimpleFlow":
        of_instance = cv2.optflow.createOptFlow_SimpleFlow()
    elif method == "SparseToDense":
        of_instance = cv2.optflow.createOptFlow_SparseToDense()

    try:
        delta = of_instance.calc(prev_frame, next_frame, None) * coef

        if method in ["Farneback", "SimpleFlow"]:
            # variational refinement
            # Name depends on version of CV2
            # delta = cv2.optflow.createVariationalFlowRefinement().calc(prev_frame, next_frame, delta)
            delta = cv2.VariationalRefinement_create().calc(prev_frame, next_frame, delta)
            delta = np.nan_to_num(delta)
            delta = fill_holes(delta)
    except:
        print("WARNING: Couldn't compute optical flow. Setting to zeros")
        shape = [data_instance.shape[1], data_instance.shape[2], data_instance.shape[0]]
        delta = np.zeros(shape)

    return delta
##########################

for nt in range(len(filelist)):
	# Load new image
	now_time = start_time + datetime.timedelta(seconds=300.*nt)
	var,file_ID,hourval,minval = user_functions.loadfile(DATA_DIR + filelist[nt])
	print(file_ID)
	write_file_ID = 'S' + sql_str + '_T'+ thr_str +'_A'+ areastr +'_'+ file_ID
	NewLabels=object_tracking.label_storms(var,minpixel,threshold,struct2d,under_t)
	# oldmask, newmask, USED FOR DERIVING (dx,dy)
	# THESE CAN BE CHANGED USING EXPERT KNOWLEDGE (e.g. use raw data rather than binary masks, if displacement information is contained in structures within objects)
	# !!! NB If raw data are used (i.e. not zeros and ones) then fftpixels needs to be changed to remain sensible !!!
	if len(OldLabels) > 1:
	    # CHECK TIME DIFFERENCE BETWEEN CONSECUTIVE IMAGES 
	    dtnow = user_functions.timediff(oldhourval,oldminval,hourval,minval)
	    num_dt = dtnow/dt
	    if dtnow > dt_tolerance:
		print('Data are too far apart in time --- Re-initialise objects')
		OldData, OldLabels, oldvar, newvar, prev_time = [], [], [], [], []
		newwas = 1
		plot_vectors = False
		continue
	    oldmask = np.where(OldLabels>=1,1,0)
	    newmask = np.where(NewLabels>=1,1,0)

        delta_x = None
        delta_y = None
        if use_new_opt_flow:
            if nt > 0:
                # Load data from previous timestep
                prev_time = start_time + datetime.timedelta(seconds=300.*(nt - 1))
	        prev_var, _, _, _ = user_functions.loadfile(DATA_DIR + filelist[nt - 1])

                # Stack data into required shape
                input_data = np.stack((prev_var, var), axis=0)
                
                # Scale input data to uint8 [0-255] 
                scaled_data, c1, c2 = scaler(input_data)

                # Calculate optical flow displacements in each direction (in pixels)
                of = calculate_of(scaled_data, method=of_method, direction=direction)
                delta_x = of[::, ::, 0]
                delta_y = of[::, ::, 1]
            
	# Call object tracking routine
	# NewData = list of objects and properties
	# newwas = final label number
	# NewLabels = array with object IDs from [1, nummax] as found by label_storms
	# newumat, newvmat = arrays with (dx,dy) displacement between two images (NB not displacement per dt!!!) 
	# wasarray = array with object IDs consistent across images (i.e. tracked IDs)
	# lifearray = array with object lifetime consistent across images
	NewData, newwas, NewLabels, newumat, newvmat, wasarray, lifearray = object_tracking.track_storms(OldData, var, newwas, NewLabels, OldLabels, xmat, ymat, fftpixels, dd_tolerance, halosq, squarehalf, oldmask, newmask, num_dt, lapthresh, misval, doradar, under_t, IMAGES_DIR, write_file_ID, flagplottest, use_new_opt_flow=use_new_opt_flow, delta_x=delta_x, delta_y=delta_y)
	# Write tracked storm information (see object_tracking.write_storms)
	if flagwrite:
		object_tracking.write_storms(write_file_ID, start_time, now_time, label_method, squarelength, rafraction, newwas, NewData, doradar, misval, IMAGES_DIR)
	# Plot tracked storm information (see user_functions.plot_example)
	if flagplot:
		user_functions.plot_example(write_file_ID, nt, var, xmat, ymat, newumat, newvmat, num_dt, wasarray, lifearray, threshold, IMAGES_DIR, plot_vectors)
	# Save tracking information in preparation for next image
	OldData = NewData
	OldLabels = NewLabels
	oldvar = var
	oldhourval = hourval
	oldminval = minval
	plot_vectors = True

