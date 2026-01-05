# simple-track
Threshold-based object tracking algorithm for 2D data

# file description

"wrapper.py" is the script that runs the tracking algorithm with the desired input and output directories and parameter choices. This should be the only Python file you need to modify. Relevant parameters are described below.

"object_tracking.py" is the Python program that actually does the tracking. This script should not need modification, unless the user wishes to store additional information about tracked objects. That additional information will need to be added to the "class StormS()" section and to the "write_storms" function in this script.

"user_functions.py" is the user-specified script to load files, calculate time differences, and plot output. Other user-specified functions should be added here.

# parameters

"wrapper.py" contains a set of parameters that all need changing in relation to the user preferences and data sets. 

Data-relevant parameters are:
* dt:    		The minimum separation in time possible between consecutive images
* dt_tolerance:	The maximum separation in time allowed between consecutive images (to allow for missing data)
* under_t:	Is the variable of interest smaller than a threshold (e.g. Brightness temperatures, under_t=True) or greater (e.g. Rainfall, under_t=False)

Tracking-relevant parameters are:
* struct2d:	Defines the neighbour-searching function, np.ones((3,3)) is 8-point connectivity.
* minpixel:	The minimum number of pixels for an object to be tracked
* squarelength:	The size in pixels of individual square regions for which displacement vectors will be calculated (should be large enough to cover several mid-sized objects)
* rafraction:	The minimum fractional cover of objects required in order for displacement vectors to be calculated
* dd_tolerance:	The maximum difference (in number of pixels) allowed between adjacent displacement vectors
* halopixel:	Radius of halo in pixels to look for orphaned objects (objects that do not overlap but that are within this radius of a "parent" will still be classed as "child" and to have spawned off the original object)
* lapthresh:	Minimum overlap fraction required for objects to be considered potentially the same between consecutive images [Default is 0.6]

Output-relevant parameters are:
* flagwrite:	If False, then no text files with object information is included in the output. [Default should be True]
* misval:		Preferred value to used for missing values.
* flagplot:	If True, a few images are included in the output (plotting function defined in "user_functions.py" [Trials should set this to True, long runs could set it to False to save time]
* flagplottest:	If True, numerous test images are included to check the displacement vector calculations [Default should be False]

# input

A function to read the data and a function to calculate time difference between consecutive files should be provided in "user_functions.py"

The directory of the input data should be specified in "wrapper.py" under DATA_DIR

Note that the corresponding (x,y) arrays should be defined in "wrapper.py" as xmat and ymat. 

Further changes may be necessary to the listing and date/time specifications in "wrapper.py"

# output

The tracking algorithm outputs a text file with a name related to the matching input filename and
containing details of the objects identified in each timestep, eg.

The number following storm/cell is the storm/cell id
* area:     the number of grid cells that met the threshold in this storm
* centroid: the centre of the rectangle around the storm or the centre of a cell [latix, lonix]
* box:      defines the rectangle around the storm [minlatix, minlonix, nlats, nlons]
* life:     the number of timesteps this storm has been seen minus 1, i.e. if this is zero the storm
          has only been seen in one image so far; this should increment in each successive timestep until the storm disappears.
		  If the storm was created by splitting from a parent, life will be the life of the parent when this storm was created
		  and will increment thereafer.
* dx,dy:       the velocity of this storm at this timestep as determined from pattern correlation
* meanv:    the average of the variable of interest for the object
* extreme:  the minimum value of the variable of interest for the object
* accreted: the storm ids that merged with this one at this timestep
* parent:   if this storm split from another storm at this timestep this is the id of the parent storm 
* child:    the ids of storms that split off this storm at this timestep.

Additional properties can be added by experienced users by editing "object_tracking.py" (see above).

Plots can be generated based on the output (e.g. in "user_functions.py" see plot_example function) but this will slow down the code significantly. 
