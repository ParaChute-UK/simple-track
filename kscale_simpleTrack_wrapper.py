import object_tracking
import numpy as np
import datetime
import os
import sys
import argparse
import user_functions

# TO RUN THIS NOW NEED jaspy/3.11/v20240302

##################################################################
# THE FOLLOWING PARAMETERS SHOULD BE CHANGED BASED ON THE DATA (RESOLUTION ETC.)
##################################################################

# Integer number (dimensions user-defined) to identify MINIMUM time difference between consecutive data files
# Example 1: Radar data 5-minutes apart with time stamp in filename, dt = 5
# Example 2: Satellite brightness temperatures hourly with time stamp in filename, dt = 1
# NB. When writing storms, (dx,dy) will have units PIXELS per TIME STEP (specified by dt), so already scaled by number of missing files
dt = 1 
dt_tolerance = 2 # Maximum separation in time allowed between consecutive images

under_t=True ## True = labelling areas *under* the threshold (e.g. brightness temperature), False = labelling areas *above* threshold (e.g. rainfall)
#Threshold taken as MCSMIP Tb = 241K constraint.
threshold = 241. ## Threshold used to identify objects (with value of variable greater than this threshold) - corresponds to -32C
minpixel = 9. #For 0.1deg control run -> ~1000km2 ## Minimum object size in pixels. 
squarelength = 100. ## Size in pixels of individual squares to run fft for (dx,dy) displacement. Must divide (x,y) lengths of the array!
rafraction = 0.01 ## Minimum fractional cover of objects required for fft to obtain (dx,dy) displacement
dd_tolerance = 3. # Maximum difference in displacement values between adjacent squares (to remove spurious values) - scaled by num_dt if necessary
halopixel = 5. ## Radius of halo in pixels for orphan storms - big halo assumes storms may spawn "children" at a distance multiple pixels away

flagwrite = True ## For writing storm history data in a text file 
doradar = False ## doradar=True is calculate range and azimuth for real-time tracking with radar (e.g. Chilbolton). doradar=False any other use, radar coordinates not relevant
misval = -999 ## Missing value
struct2d = np.ones((3,3)) ## np.ones((3,3)) is 8-point connectivity for labelling storms. Can be changed to user preference.

flagplot = False ## For plotting images (vectors and IDs). Also set plot_type...
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

#label_method='Rainfall rate > ' + thr_str + 'mm/hr'    
label_method="Tb < "+thr_str+"K"

##################################################################
# THE REMAINDER IS THE SET UP FOR THE EXAMPLE DATA
# THIS SHOULD BE ADJUSTED RELEVANT TO THE USER DATA
# !!!TEST: MAKE SURE xall AND yall DIVIDE IN squarelength!!!
##################################################################
# Make sure appropriate to data!



parser = argparse.ArgumentParser()
parser.add_argument("-r", "--region", required=True, type=str)
parser.add_argument("-d", "--domain", required=False, default="channel", type=str)
parser.add_argument("-res", "--res", required=False, default="n2560", type=str)
parser.add_argument("-c", "--config", required=False, default="RAL3p2", type=str)
args = parser.parse_args()

region=args.region.lower()
domain=args.domain.lower()
res_name=args.res
config=args.config

dymnd_run="20160801T0000Z"
#dymnd_run="20200120T0000Z"

#######################################################################################################
# DATA SPECIFIC SETTINGS - ALL PATHS AVAILABLE TO JASMIN kscale gws USERS (contact: Huw Lewis)

if domain!="lam":
	# Path for all analysis grid (common, 0.1deg) CTC data
	root='/gws/nopw/j04/kscale/DATA/outdir_{}/DMn1280GAL9/{}_{}_{}/'.format(dymnd_run,domain,res_name,config)
	if region == "ea-waf":
		# SPANS 0 - 25 N, -40 - 30 W.
		xmat,ymat = np.meshgrid(range(700),range(250))
	elif region == "summer" or region=="winter":
		xmat,ymat = np.meshgrid(range(3600),range(600))
else:
	# Multiple LAMs available in hierarchy
	# Africa LAM [LAM2.2 IN MAYBEE ET AL]
	if region == "wafrica":
		if config=="RAL3p2_RainEvapOff":
			root='/gws/nopw/j04/kscale/DATA/outdir_{}/DMn1280GAL9/lam_africa_km2p2_RAL3p2_RainEvapOff/'.format(dymnd_run)
		else:
			root='/gws/nopw/j04/kscale/DATA/outdir_{}/DMn1280GAL9/lam_africa_{}_{}/'.format(dymnd_run,res_name,config)
		# CRITICL TO REMEMBER: wafrica HERE LARGER DOMAIN THAN IN SNAPSHOT etc CODE, TO ENABLE YMAT DIVISIBILITY. SPANS -6 - 24 N, -18 - 32 W.
		xmat,ymat = np.meshgrid(range(0,300),range(0,500))

######
xall = np.size(xmat,0) # Only used to check grid dimensions
yall = np.size(ymat,1) # Only used to check grid dimensions
if (np.fmod(xall,squarelength)!=0 or np.fmod(yall,squarelength)!=0):
	raise ValueError('Your grid does not match a multiple of squares as defined by squarelength')
    
#######################################################################################################
#0.1deg post-processed kscale output
DATA_DIR = root+'/single_olwr/'
IMAGES_DIR = f"/gws/nopw/j04/kscale/USERS/bmaybee/simpleTrack_MCS_outputs/{domain}_{res_name}_{config}_{region}/"
file_str = '20160801T0000Z_{}_olwr_hourly.nc'.format(domain)
#################################################################

if not os.path.isdir(IMAGES_DIR):
	os.makedir(IMAGES_DIR)

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

rst=False
#tools for restart from crash if necessary - change above variable!
last_storm_file_ID=[]
if rst:
    #Extra factor of -1 to account for indexing offset
	rst=len(os.listdir(IMAGES_DIR))
	last_storm_file_ID.append(os.listdir(IMAGES_DIR)[-1])
print(last_storm_file_ID)

#start_time = datetime.datetime(2012,8,25,14,5,0,0)
start_time = datetime.datetime(2016,8,1,0,0,0,0)
num_days = 40
oldhourval = []
oldminval = []
oldmask = []
newmask = []
num_dt = []

for nt in range(rst,num_days*24):
	# Load new image
	#print(OldLabels[0].shape)
	now_time = start_time + datetime.timedelta(hours=nt)
	#try:
	var,file_ID,hourval,minval = user_functions.loaddayfile(DATA_DIR,file_str,now_time,region=region)
	#except:
	#	print("Problem with input file, ",tstamp)
	#	continue
	#For BT based detection: convert OLR to Tb field. Crop may be necessary to ensure squarelegnth divides grid dimensions.
	var=user_functions.tb_from_olr(var)

	if nt==rst and (len(last_storm_file_ID) == 1): # Continue from existing file if it exists
		print("Continuing from previous file")
		last_storm_file_ID = last_storm_file_ID[0]
		#last_label_file_ID = last_label_file_ID[0].replace('.npy', '')
		OldData, start_time, newwas = object_tracking.read_storms(IMAGES_DIR + last_storm_file_ID, under_t=under_t, extra_thresh=[])
		OldLabels = object_tracking.label_storms(var,minpixel,threshold,struct2d,under_t)
		oldvar = var	
		oldhourval = hourval
		oldminval = minval
		continue 
	    
	print(file_ID)
	write_file_ID = 'S' + sql_str + '_T'+ thr_str +'_A'+ areastr +'_'+ file_ID
	NewLabels=object_tracking.label_storms(var,minpixel,threshold,struct2d,under_t)
	# oldmask, newmask, USED FOR DERIVING (dx,dy)
	# THESE CAN BE CHANGED USING EXPERT KNOWLEDGE (e.g. use raw data rather than binary masks, if displacement information is contained in structures within objects)
	# !!! NB If raw data are used (i.e. not zeros and ones) then fftpixels needs to be changed to remain sensible !!!
	if len(OldLabels) > 1:
	    # CHECK TIME DIFFERENCE BETWEEN CONSECUTIVE IMAGES 
		dtnow = user_functions.timediff(oldhourval,oldminval,hourval,minval)/60
		num_dt = dtnow/dt
		if dtnow > dt_tolerance:
			print('Data are too far apart in time --- Re-initialise objects')
			OldData, OldLabels, oldvar, newvar, prev_time = [], [], [], [], []
			newwas = 1
			plot_vectors = False
			continue
		oldmask = np.where(OldLabels>=1,1,0)
		newmask = np.where(NewLabels>=1,1,0)
	# Call object tracking routine
	# NewData = list of objects and properties
	# newwas = final label number
	# NewLabels = array with object IDs from [1, nummax] as found by label_storms
	# newumat, newvmat = arrays with (dx,dy) displacement between two images (NB not displacement per dt!!!) 
	# wasarray = array with object IDs consistent across images (i.e. tracked IDs)
	# lifearray = array with object lifetime consistent across images

####
	#NewData, newwas, NewLabels, newumat, newvmat, wasarray, lifearray = object_tracking.track_storms(OldData, var, newwas, NewLabels, OldLabels, xmat, ymat, fftpixels, dd_tolerance, halosq, squarehalf, oldmask, newmask, num_dt, lapthresh, misval, doradar, under_t, IMAGES_DIR, write_file_ID, flagplottest)
	try:
		NewData, newwas, NewLabels, newumat, newvmat, wasarray, lifearray = object_tracking.track_storms(OldData, var, newwas, NewLabels, OldLabels, xmat, ymat, fftpixels, dd_tolerance, halosq, squarehalf, oldmask, newmask, num_dt, lapthresh, misval, doradar, under_t, IMAGES_DIR, write_file_ID, flagplottest)
	# Write tracked storm information (see object_tracking.write_storms)
	except:
		print("Failed to track")
		NewData, newwas = OldData, newwas
	if flagwrite:
		object_tracking.write_storms(write_file_ID, start_time, now_time, label_method, squarelength, rafraction, newwas, NewData, doradar, misval, IMAGES_DIR)
	# Plot tracked storm information (see user_functions.plot_example)
	if flagplot:
		try:
			user_functions.plot_example(write_file_ID, nt, var, xmat, ymat, newumat, newvmat, num_dt, wasarray, lifearray, threshold, IMAGES_DIR, plot_vectors)
		except:
			print("Can't output")
			pass
	# Save tracking information in preparation for next image
	OldData = NewData
	OldLabels = NewLabels
	oldvar = var
	oldhourval = hourval
	oldminval = minval
	plot_vectors = True
