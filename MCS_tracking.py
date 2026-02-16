# python
import numpy as np
import glob
import iris
import sys
import pickle
import datetime as dt
import scipy.ndimage
import cv2
from iris.coord_categorisation import add_categorised_coord
import track_period_dicts as tpd



def add_hour_of_day(cube, coord, name='hour_of_day'):
    add_categorised_coord(cube, name, coord,
                          lambda coord, x: coord.units.num2date(x).hour)

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

        #
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
#         of_instance = cv2.optflow.createOptFlow_DIS()
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
    #         delta = cv2.optflow.createVariationalFlowRefinement().calc(prev_frame, next_frame, delta)
            delta = cv2.VariationalRefinement_create().calc(prev_frame, next_frame, delta)
            delta = np.nan_to_num(delta)
            delta = fill_holes(delta)
    except:
        print("WARNING: Couldn't compute optical flow. Setting to zeros")
        shape = [data_instance.shape[1], data_instance.shape[2], data_instance.shape[0]]
        delta = np.zeros(shape)

    return delta


def advect_previous_labels(prev_labels, Prev_Storms,prev_Tb, uwind,vwind,time_res_mins,gridlength,dynamic_maxdist=False, verbose=False):
    '''
    :param prev_labels: numpy array of storm labels from the previous timestep
    :param Prev_Storms: List of storm objects from previous timestep
    :param pre_Tb: Numpy array of previous Tb data for advecting
    :param uwind: Numpy array of x winds, ms-1
    :param vwind: Numpy array of y winds, ms-1
    :param time_res_mins: int value of time resolution in minutes
    :param gridlength: float of horizontal grid resolution
    :return:
    '''

    # The uv storm vector is only relevant for determining max distance threshold exceedances - applying mean u and mean v to each pixel is correct.
    #TODO - WK - Bit of duplication to advect labels AND old Tb (needed for refl-wghting). Possible simplification to advect just R and identify labels in that field?

    if verbose:
        print('Advecting previous labels...')

    advected_label_clash_counter=0 # to count % of clashes

    # Convert wind field [ms-1] to number of grid boxes per timestep for advection calcs.
    uwind_pixels = ((uwind * 60) * time_res_mins) / (gridlength * 1000) # u pixel component
    vwind_pixels = ((vwind * 60) * time_res_mins) / (gridlength * 1000) # v pixel component
    uv_wind_pixels = np.sqrt(uwind_pixels**2 + vwind_pixels**2) # wind field vector

    advected_labels = np.zeros(prev_labels.shape)  # Initialise advected labels array
    advected_Tb = np.zeros(prev_labels.shape)  # Initialise advected Tb array - for calculating reflectivity-centroid weighting if asked for
    xmat,ymat =  np.meshgrid(range(advected_labels.shape[1]), range(advected_labels.shape[0])) #grid for calculating distances

    for ns in range(len(Prev_Storms)): # Loop over each storm from previous time step
        jj = Prev_Storms[ns].storm # previous storm label
        labelind = np.where(prev_labels == jj)  # Indices of previous storm jj
        u = np.mean(uwind_pixels[labelind])  # Mean of all u shifts for that storm (in pixels)
        v = np.mean(vwind_pixels[labelind])  # Mean of all v shifts for that storm (in pixels)
        # C.SHORT think this is better defined as below
#         uv = np.mean(uv_wind_pixels[labelind]) # vector pixel shift of storm (in pixels)
        uv = np.sqrt(u**2 + v**2) # translation speed of storm (pixels per timestep)

        #================= Apply cap on max speed allowed (given by optical flow) ==========##
        if dynamic_maxdist ==True:    # Use Han et al 2009 recommended variable thresholds.
            # Note - does not assume large cells move more, just allows random centroid displacements to be tolerated.
            prev_storm_area = Prev_Storms[ns].area*gridlength**2
            if prev_storm_area<=300:
                max_cell_speed_kmh = 100.
            elif prev_storm_area>300 and prev_storm_area<=500:
                max_cell_speed_kmh = 150.
            elif prev_storm_area>500:
                max_cell_speed_kmh = 200.
        else:
            max_cell_speed_kmh=999

        max_speed_pixels = (max_cell_speed_kmh / gridlength) * (time_res_mins / 60)

        if uv>max_speed_pixels: # Don allow storm to advect as exceeding max distance threshold
            u=0.
            v=0.
            print(f'WARNING: storm label {jj} exceeded max. distance allowed for its size!! ')
        if verbose:
            print(f'Applying shift (in grid boxes) u={np.round(u,2)}, v={np.round(v,2)} to storm label {jj}')
        if u == 0.0 and v == 0.0: # Advected labels are the same as previous labels if no shift applied (TODO - chance of averaging zero while pixels moving??)
            advected_labels[labelind] = jj #
            advected_Tb[labelind] =  prev_Tb[labelind]# just same as before
            if verbose:
                print(f'No u and v shifts applied; advected label just {jj}')
        else:
            # Used to be additional if statement here for implement a max distance storm can move condition. More to cater for old wind calc issues?
            for ii in range(np.size(labelind, 1)):  # WK - For each pixel in storm
                if verbose:
                    print_count =1 # only want to print advected outside of the domain once
                # C.SHORT are we convinced x,v and y,u are the right way around here?
                newxind = labelind[0][ii] + int(np.around(v))  # assign new x indices for storm (v shifts applied to first indices; numpy indexing)
                newyind = labelind[1][ii] + int(np.around(u))  # assign new y indices for storm  (u shifts applied to first indices; numpy indexing)

                if newxind > np.size(advected_labels, 0) - 1 or newyind > np.size(advected_labels, 1) - 1 or newxind < 0 or newyind < 0:  # Edge case - if new labels within the domain, continue
                    if verbose:
                        if print_count==1:
                            print(f'At least one pixel of storm {jj} is advected outside of the domain, skipping to next storm')
                    continue # Next storm.
                elif advected_labels[newxind, newyind] > 0:  # Edge case - if new idcs do not leave the domain, but advected label already exists, deal with here
                    advected_label_clash_counter=advected_label_clash_counter+1
                    if verbose:
                        print(f'A storm (id={advected_labels[newxind, newyind]}) has already been advected here (x={newxind},y={newyind})')
                    nq = int(advected_labels[newxind, newyind] - 1) # what is the index of the other storm that exist here
                    olddist = (xmat[newxind, newyind] - Prev_Storms[nq].centroidx) ** 2 + (ymat[newxind, newyind] - Prev_Storms[nq].centroidy) ** 2  # WK - distance between competing unadvected storm centre and current pixel
                    newdist = (xmat[newxind, newyind] - Prev_Storms[ns].centroidx) ** 2 + (ymat[newxind, newyind] - Prev_Storms[ns].centroidy) ** 2  # WK - distance between current unadvected storm centroid and current pixel
                    if newdist < olddist:  # If distance new storm centroid closer to current image than the current storm centroid is to the previous image, new label is just the old one.
                        advected_labels[newxind, newyind] = jj
                        advected_Tb[newxind, newyind] = prev_Tb[labelind[0][ii],labelind[1][ii]] # the original value at the advected locations
                        if verbose:
                            print('Pixel closest to current unadvected storm centroid so is chosen for advected label')
                    else:
                        if verbose:
                            print('Original unadvected centroid closest to pixel in question - advected label is retained.')

                else:  # Non-zero advection and advected labels do not already exist at the new indices
                    advected_labels[newxind, newyind] = jj
                    advected_Tb[newxind, newyind] = prev_Tb[labelind[0][ii],labelind[1][ii]] # the original value at the advected locations

    advected_label_clash_percent = (advected_label_clash_counter/np.sum(prev_labels>0))*100

    return advected_labels, advected_Tb, advected_label_clash_percent




def assign_historical_tracks_to_current_storm(Curr_Storms,Prev_Storms, ns, chosen_overlap_label):
    '''
    :param Curr_Storms: List of Current Storm Objects
    :param Prev_Storms: List of Previous Storm Objects
    :para ns: Index of Current Storm object being considered
    :param chosen_overlap_label: Storm label of previous storm to assign tracks to current storm
    :return:
    '''

    # Lot's of duplication here, sure can be improved (e.g. the treating ii=0 and else differently...). Can improve later

    # Curr_Storms[ns].was = Prev_Storms[chosen_overlap_label].was  # # Current storm was label given Prev Storm was label
    # C.SHORT Now using previous storm ID as the was label, rather than inheriting the was label
    Curr_Storms[ns].was = Prev_Storms[chosen_overlap_label].storm  # # Current storm was label given Prev Storm ID
    Curr_Storms[ns].life = Prev_Storms[chosen_overlap_label].life + 1  # Assign previous life value to current storm, +1

    if verbose:
        print(f"Trying to assign track history to curr storm ns = {ns}, chosen_overlap_label = {chosen_overlap_label}")
        print(f"Prev storm [{chosen_overlap_label}] life = {Prev_Storms[chosen_overlap_label].life}")
    if Prev_Storms[chosen_overlap_label].life > 1: # If previous storm life longer than 1 timestep
        for ii in range(Prev_Storms[chosen_overlap_label].life): # Need to loop over each timestep and append
            if ii == 0: # If first timestep, start track lists
                Curr_Storms[ns].track_xpos = [Prev_Storms[chosen_overlap_label].track_xpos[ii]]
                Curr_Storms[ns].track_ypos = [Prev_Storms[chosen_overlap_label].track_ypos[ii]]
                Curr_Storms[ns].track_lonpos = [Prev_Storms[chosen_overlap_label].track_lonpos[ii]]
                Curr_Storms[ns].track_latpos = [Prev_Storms[chosen_overlap_label].track_latpos[ii]]
            else: # Append subsequent positions to this lise
                Curr_Storms[ns].track_xpos.append(Prev_Storms[chosen_overlap_label].track_xpos[ii])
                Curr_Storms[ns].track_ypos.append(Prev_Storms[chosen_overlap_label].track_ypos[ii])
                Curr_Storms[ns].track_lonpos.append(Prev_Storms[chosen_overlap_label].track_lonpos[ii])
                Curr_Storms[ns].track_latpos.append(Prev_Storms[chosen_overlap_label].track_latpos[ii])

        # Once added historical track info, add current timestep position to this track
        Curr_Storms[ns].track_xpos.append(Curr_Storms[ns].centroidx)
        Curr_Storms[ns].track_ypos.append(Curr_Storms[ns].centroidy)
        Curr_Storms[ns].track_lonpos.append(Curr_Storms[ns].centroidlon)  # WK - added for map plotting
        Curr_Storms[ns].track_latpos.append(Curr_Storms[ns].centroidlat)  # WK - added for map plotting

    else: # Else if previous storm only has lifetime of 1, just start list and then append current position
        Curr_Storms[ns].track_xpos = [Prev_Storms[chosen_overlap_label].track_xpos]
        Curr_Storms[ns].track_ypos = [Prev_Storms[chosen_overlap_label].track_ypos]
        Curr_Storms[ns].track_xpos.append(Curr_Storms[ns].centroidx)
        Curr_Storms[ns].track_ypos.append(Curr_Storms[ns].centroidy)

        Curr_Storms[ns].track_lonpos = [Prev_Storms[chosen_overlap_label].track_lonpos]  # WK - added for map plotting
        Curr_Storms[ns].track_latpos = [Prev_Storms[chosen_overlap_label].track_latpos]  # WK - added for map plotting
        Curr_Storms[ns].track_lonpos.append(Curr_Storms[ns].centroidlon)  # WK - added for map plotting
        Curr_Storms[ns].track_latpos.append(Curr_Storms[ns].centroidlat)  # WK - added for map plotting

    return Curr_Storms


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


def precip_to_mm(cube, freq = 'day'):
    '''
    What it does  : Convert Precip Units to mm/day
    How to use it : precip_to_mmpday(cube)
    Example       : cube = precip_to_mmpday(cube)
    Who           : Rob Chadwick, Added by Dave Rowell
    Modified      :
    :param str freq: day or 3h for example
    '''

    if ( cube.units in ['kg m-2 s-1', 'kg m-2'] ):
        rho = iris.coords.AuxCoord(1000.0,long_name='ref density',units='kg m-3')
        outcube = cube/rho # converts to ms-1 so that can use convert units
        if cube.units == 'kg m-2 s-1':
            outcube.convert_units(f'mm ({freq})-1')
        else:
            outcube.convert_units('mm')
            outcube.units = f'mm ({freq})-1'
        outcube.rename(cube.name())
    elif ( cube.units == 'm s-1' ):
        outcube = cube
        outcube.convert_units(f'mm {freq}-1')
        outcube.standard_name = cube.standard_name
    elif ( cube.units in ['mm 3h-1', 'mm (3h)-1'] ):
        outcube = cube
        outcube.convert_units(f'mm {freq}-1')
        outcube.standard_name = cube.standard_name
        warnings.warn(f'Converted units from mm (3h)-1 to {outcube.units}')
    elif ( cube.units in ['mm/3h', 'mm/3hr'] ):
        outcube = cube
        outcube.units = 'mm (3h)-1'
        outcube.convert_units(f'mm {freq}-1')
        outcube.standard_name = cube.standard_name
        warnings.warn(f'Converted units from mm (3h)-1 to {outcube.units}')
    elif ( cube.units in [f'mm {freq}-1', 'mm', f'mm ({freq})-1'] ):
        outcube = cube
    elif ( cube.units == 'mm/90days' ):
        outcube = cube
    else:
        raise ValueError(f'cube not in mm/{freq}, kg/m2/s or m/s ', cube.units)

    return outcube



def callback_overwrite(cube, field, filename):
    coord2rm = ['forecast_reference_time', 'forecast_period', 'season_number']
    for co2rm in coord2rm:
        if co2rm in [coord.name() for coord in cube.coords()]:
            cube.remove_coord(co2rm)
    attributes_to_overwrite = ['date_created', 'log', 'converter', 'um_streamid',
                               'creation_date', 'history', 'iris_version', 'prod_date',
                               'CDI', 'CDO', 'Conventions','ArchiveMetadata.0', 'CoreMetadata.0','BeginTime',
                               'EndTime','BeginDate','EndDate','FileHeader','InputPointer','ProductionTime','um_version']
    for att in attributes_to_overwrite:
        try: # try statement replaces old has_key loop (removed in latest iris?)
            cube.attributes[att] = 'overwritten'
        except:
            continue

    attributes_to_del = ['radar.flags', 'log', 'iris_version']
    for att in attributes_to_del:
        try:# try statement replaces old has_key loop (removed in latest iris?)
            del cube.attributes[att]
        except:
            continue
    if cube.coords('T'):  # for GPCP
        cube.coord('T').standard_name = 'time'




# This function is used to identify and label Tb objects in the scene
def label_objects(data, minarea, threshold, struct):
    import scipy.ndimage as ndimage
    """
    :param data: Numpy array
    :param minarea: Min area of objects
    :param threshold: Threshold that identifies objects
    :param struct: How objects are connected.
    :return:
    """
    binR = np.zeros_like(data)  # initialise an array of zeros that is the same as R
    binR[np.where(data <= threshold)] = 1 # Set all regions below threshold to 1. Note this has been inverted for Tb
    binR[np.where(data > threshold)] = 0 # Set all regions above threshold to 0
    id_regions, num_ids = ndimage.label(binR, structure=struct) # Identify objects (contigous regions above threshold)

    id_sizes = np.array(ndimage.sum(binR, id_regions, range(num_ids + 1))) # Sizes of each object
    area_mask = (id_sizes < minarea) # Mask to apply to remove objects that are below minarea
    id_sizes = id_sizes[id_sizes >= minarea] # Creates list of all identified object sizes with area greater than minarea
    binR[area_mask[id_regions]] = 0 # Apply mask
    id_regions, num_ids = ndimage.label(binR, structure=struct) # Re-identify objects now those that are too small removed
    print('num_ids = ', num_ids)
    return id_regions


def zoom_into_reg(cube,reg):
    acube = cube.copy()
    lon_extent = iris.coords.CoordExtent(acube.coord('longitude'), reg['lons'][0], reg['lons'][1] )
    lat_extent = iris.coords.CoordExtent(acube.coord('latitude'), reg['lats'][0], reg['lats'][1] )
    zoomed_cube = acube.intersection(lon_extent,lat_extent)
    return zoomed_cube


def convert_Tf_to_tb(tf):
    """Convert the Tb to brightness temperature."""
    # Calculate the equivalent temp flux
    # Calculate the brightness temperature
    a = 1.228
    b = -1.106e-3
    tb_var = (-a + np.sqrt(a ** 2 + 4 * b * tf)) / (2 * b)
    return tb_var


def apply_erosion_dilation(labels, rain, iterations=1):
    """
    Erodes and then dilates labels to break bridges between objects that result in "false mergers"
    :return:
    :param labels:2D numpy array of object labels
    :param labels: 2D numpy rain field to re-label after erosion and dilation
    :return: 2D numpy array of rain with fa
    """

    import cv2
    dilation_erosion_kernel = np.ones([5, 5])
    eroded_labels = cv2.erode(src=labels.astype('int16'), kernel=dilation_erosion_kernel, iterations=iterations) # wont work if int32!
    dilated_labels = cv2.dilate(src=eroded_labels, kernel=dilation_erosion_kernel, iterations=iterations)

    return dilated_labels


if __name__ == '__main__':

    # Currently this takes just under 3 hours to run... This means there is plenty of scope to add other vars and still
    # fit on the normal queues. Possible options include:
    # Max vertical velocity of storm
    # Wind shear based on u,v at two different levels (2D field)
    # Mean/max echo top height from 3D radar Z
    # Hail parameters
    # More complicated, inflow CAPE/CIN etc, helicity
    # Lightning output. When in lifecycle most? Before or after a split/merge? Or initiation. Density per size?


    # ===============================================================
    timestamp_start = dt.datetime.now()
    print("****** @ ", timestamp_start, ": Starting processing... **************")

    seasons = ['djf','mam','jja','son']
    reg = dict(lons=(-25, 55), lats=(-40, 40))  # Zoom into full Africa domain
    reg_smaller = dict(lons=(-24, 54), lats=(-39, 39))  # Zoom in after labelling to ignore boundary effects/variable resolution rim

    test=False

    if test==False:
        arglist = sys.argv
        print(arglist[1:])
        nargs = len(arglist) - 1
        # Command-line arguments provided.
        print("Command-line arguments provided:", arglist)

        sm_width_pixels = int(arglist[1])
        halo_pixels = int(arglist[2])
        solve_false_merger = arglist[3].lower()=="true"
        dynamic_maxdist = arglist[4].lower()=="true"
        reflectivity_weight_centroids= arglist[5].lower()=="true"  # If using reflectivity weighed centroids
        datestr=str(arglist[6])
        threshold=float(arglist[7])
        dataset=str(arglist[8])
        min_area=int(arglist[9])
        sys.stdout.flush()

    else:
        threshold=240
        sm_width_pixels= 5
        halo_pixels=0
        solve_false_merger= False
        dynamic_maxdist= False
        reflectivity_weight_centroids=False
        datestr = "20230601"
        dataset="u-df283"
        min_area=40000

    regrid_to_GPM_IMERG=True
    if sm_width_pixels>0:
        remove_boundary_objects=False # Has to be True as boundary will have an object which want to remove.
    else:
        remove_boundary_objects=False

    hard_start_date=  tpd.suite_date_ranges[dataset]["start_date"] # Just the first data. Originally was for custom ranges

    print("sm_width_pixels:",sm_width_pixels)
    print("halo_pixels:",halo_pixels)
    print("solve_false_merger:",solve_false_merger)
    print("dynamic_maxdist:",dynamic_maxdist)
    print("threshold",threshold)
    print("min area",min_area)

    #############################################################
    #######============ Set manual settings =========##############
    #############################################################

    # Alays optical flow winds
    use_wind_data=False
    wind_label= 'of_winds'

    #matplotlib.use('Qt5Agg')
    if regrid_to_GPM_IMERG:
        gridlength = 10. # GPM MERGIR is 10km.
    else:
        gridlength = 4. # GPM MERGIR is 10km.

    # Some varaibles still set manually here
    of_method = "DIS"
    direction = "backward"
    time_res_mins = 60
    min_area_pixels = int(min_area / gridlength / gridlength)  # WK - Fed into label code for min # grid points to label (corresponding to at least minarea)
    minareastr = str(int(min_area))
    Tbstr = f'{int(threshold)}K'

    verbose=False

    if time_res_mins==15:
        expected_delta_t=900 # seconds, for some reason minutes not available?
    elif time_res_mins==60:
        expected_delta_t=3600 # seconds, for some reason minutes not available?

    ######################
    ##################################################################
    # SET UP OF DISCRETE VALUES BASED ON USER INPUT PARAMETERS
    ##################################################################
    np.set_printoptions(suppress=True) # just to make reading print output easier

    if use_wind_data == True:  # If using model winds, save to different directory for comparison.
        wind_label = 'model_winds'
    else:
        wind_label= 'of_winds'

    if reflectivity_weight_centroids== True:  # If using refrlecitivty weighed centroids
        centroid_label = "rwc" # reflectivity-weighted centroids
    else:
        centroid_label = "geom" # reflectivity-weighted centroids

    if solve_false_merger:
        false_merger_label="removed_false_mergers"
    else:
        false_merger_label="incl_false_mergers"

    halokm =halo_pixels*gridlength # number of pixels times resolution
    halo = int(halokm / gridlength)  ## Number of pixels
    halosq = halo ** 2 # WK halosq only thing that's used in halo code below
    pad_label=f"halo_{halokm}km"
    pad_label=f"halo_{halo_pixels}pixels"

    if dynamic_maxdist == True:  # If using model winds, save to different directory for comparison.
        dynamic_label = 'dst'
    else:
        dynamic_label= 'fixed'

    #################################################################
    ##########====== Deal with input time requests ====== #########
    #################################################################

    startyyyy=str(datestr[:4])
    startmm=str(datestr[4:6])
    startdd=str(datestr[6:8])

    print("")
    print(f"TRACKING STORMS IN Tb FOR {datestr}")
    print("")

    # We want there to be single timestep overlap between days so can stitch tracks together (allowing parallelisation)

    if dataset in ["u-cw282", "u-cx129","u-cw288","u-cx047"]:  # Then these use a 360 day calendar
        first_day = cftime.Datetime360Day(int(startyyyy), int(startmm), int(startdd),0,0)
    elif dataset in ["u-dc272", "u-dc178", "u-dc139", "u-dc077"]:  # Then these use 365 day calendar (no leap years).
        first_day = cftime.DatetimeNoLeap(int(startyyyy), int(startmm), int(startdd),0,0)
    else:
        first_day = datetime(int(startyyyy), int(startmm), int(startdd), 0, 0)  # Get the first day specified.

    if startyyyy + startmm + startdd == hard_start_date:  # If this is the first date of the run, then won't be a 23:45 the day before
        start_datetime = first_day
        full_date_range=[first_day.strftime('%Y%m%d')]
    else:
        start_datetime = (first_day - timedelta(minutes=60))  # Use time delta to take a day off
        prev_day = (first_day - timedelta(days=1)).strftime('%Y%m%d')       # Use time delta to take a day off
        full_date_range=np.append(prev_day, first_day.strftime('%Y%m%d'))  # Create new daterange including day-1 to loop over

    print("Full date range to load:", full_date_range)

    # Extract the correct time window, for Tb is last hour of prev day to end of the input day
    start_partial_datetime=iris.time.PartialDateTime(start_datetime.year,start_datetime.month,start_datetime.day,start_datetime.hour,start_datetime.minute)
    end_partial_datetime=iris.time.PartialDateTime(int(startyyyy), int(startmm), int(startdd),23,30)
    datelims = [start_partial_datetime, end_partial_datetime]
    time_con = iris.Constraint(time=lambda t: datelims[0]<= t.point and datelims[-1]>= t.point)

    sys.stdout.flush()

    OUT_DIR = f'/data/scratch/william.keat/MCS_tracking_fd/DYMECS/output/{dataset}/Tb/{Tbstr}/{min_area}km2/{datestr}/{dynamic_label}/{centroid_label}_cen/{time_res_mins}min/sm_{sm_width_pixels}pixels/{pad_label}/{false_merger_label}/on_GPM_{regrid_to_GPM_IMERG}/'
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)


    if 1:
        ###############################################################################
        # SETUP (X,Y) GRID
        # load in ref cube, from which we will initialise domain specific properties. Arbitrary file used (0)
        if dataset == "GPM":
            Tb_filepath = '/data/users/regclim/transferred_data/obs_data/GPM_MERGIR/Africa/'
            ref_cube = iris.load(f'{Tb_filepath}/{startyyyy}/merg_{startyyyy}{startmm}{startdd}01_4km-pixel.nc')[0][0]
        else:
            ref_file = sorted(glob.glob(f'/data/scratch/william.keat/NEW_CP4A/hourly/{dataset}/pr/*'))[0]  # get first file of list of valid files
            ref_cube = iris.load(ref_file)[0][0]
        ref_cube = zoom_into_reg(ref_cube, reg)

        if regrid_to_GPM_IMERG == True:
            gpm_ref_cube = iris.load_cube('/data/users/gpm_imerg/production/current_version/2016/gpm_imerg_production_V07B_20160701.nc')[0]
            slightly_bigger_region = dict(lons=(-26, 56), lats=(-41, 41))  # 1 degree bigger; no missing data when zooming...
            gpm_ref_cube = zoom_into_reg(gpm_ref_cube, slightly_bigger_region)  # slightly bigger than pan-Africa domain

            # Need to prep the coords before regridding
            ref_cube.coord("latitude").coord_system = gpm_ref_cube.coord("latitude").coord_system
            ref_cube.coord("longitude").coord_system = gpm_ref_cube.coord("longitude").coord_system
            ref_cube.coord("latitude").guess_bounds()
            ref_cube.coord("longitude").guess_bounds()
            gpm_ref_cube.coord("latitude").guess_bounds()
            gpm_ref_cube.coord("longitude").guess_bounds()

            ref_cube = ref_cube.regrid(gpm_ref_cube, iris.analysis.AreaWeighted())

        smaller_reg_for_xmat = zoom_into_reg(ref_cube, reg_smaller)
        nlats = len(ref_cube.coord("latitude").points)
        nlons = len(ref_cube.coord("longitude").points)

        lonmat, latmat = np.meshgrid(smaller_reg_for_xmat.coord("longitude").points, smaller_reg_for_xmat.coord("latitude").points)  # WK: making lat/lon arrays to subset for tracks..lonmat and latmax replace ymat and xmat when subbing centroid centre for tracks
        xmat, ymat = np.meshgrid(range(lonmat.shape[0]), range(lonmat.shape[1]))  # grid to

        # For some reason, long are lons+360... Need to subtract off for true longitudes (regional subsetting later).
        ###############################################################################

        ##################################################################
        # THE FOLLOWING PARAMETERS CAN BE CHANGED, BUT SHOULD NOT BE
        ##################################################################
        misval = -999  ## Missing value
        lapthresh = 0.6  ## Minimum fraction of overlap (0.6 in TITAN)
        struct2d = np.ones((3, 3))  ## Dimension of isolated cell area for labelling

        #########################################################################################
        ###############===============  SET ADDITIIONAL PARAMETERS  ====================#########
        #########################################################################################
        plot_demo_figures = False
        plot_summary_figures = False
        plot_timestep = False

        ###################################################################
        # SET UP OF EMPTY STRUCTURE AND START TRACKING
        ###################################################################
        class StormS:
            storm = []
            area = []
            centroidx = []
            centroidy = []
            centroidlon = []  # WK - added so that can plot lat/lon tracks
            centroidlat = []  # WK - added so that can plot lat/lon tracks
            was = []
            life = []
            track_xpos = []
            track_ypos = []
            u = [misval]
            v = [misval]
            maxTb = []
            meanTb = []
            minTb = []
            maxpr = []
            meanpr = []
            minpr = []
            parent = []
            child = []
            overlap_area_with_chosen_advected_storm = []
            accreted = []
            cell = []

        Prev_Storms = []  # Initialise Prev Storms list before starting tracking

        # C.SHORT
        # safe_was_label = 1  # new 'was' label that ensures unique labels always used. Iterated over.

        ######################################################################################################
        #####=========       LOAD IN Tb DATA FOR SPECIFIED DAY RANGE (DEFAULT SHOULD BE SINGLE DAY) =====####
        ######################################################################################################
        CL = iris.cube.CubeList([])
        pr_CL = iris.cube.CubeList([])

        for nd, date2load in enumerate(full_date_range):  # Loop over all days in range
            print(date2load)
            # Need to get correct day of year for file dir
            yyyy = str(date2load[:4])
            mm = str(date2load[4:6])
            dd = str(date2load[6:8])

            if dataset == "GPM":  # Load in observed GPM-IMERg rainfall and Tb from GPM_MERGIR
                GPM_version = "V07B"
                Tb_filepath = '/data/users/regclim/transferred_data/obs_data/GPM_MERGIR/Africa/'
                pr_filepath = '/data/users/gpm_imerg/production/current_version/'

                # load pr_cube
                day_rain_cube = iris.load(f'{pr_filepath}{yyyy}/gpm_imerg_production_{GPM_version}_{date2load}.nc')[0]
                print(f'loading {pr_filepath}{yyyy}/gpm_imerg_production_{GPM_version}_{date2load}.nc')
                # Make hourly mean
                add_hour_of_day(day_rain_cube, "time")
                day_rain_cube = day_rain_cube.aggregated_by("hour_of_day", iris.analysis.MEAN)  # make hourly to match model
                day_rain_cube = zoom_into_reg(day_rain_cube, reg)  # pan Africa domain
                pr_CL.append(day_rain_cube)

                day_Tb_files = sorted(glob.glob(f'{Tb_filepath}/{yyyy}/merg_{yyyy}{mm}{dd}*_4km-pixel.nc'))
                for hour_file in day_Tb_files:
                    print(f'loading {hour_file}')
                    daily_Tb_cube = iris.load_cube(hour_file, callback=callback_overwrite)[1]  # index 1 is XX:30. This is for a closer match to hourly mean from GPM precip
                    daily_Tb_cube = zoom_into_reg(daily_Tb_cube, reg)
                    CL.append(daily_Tb_cube)

            else:
                OLR_timefreq = "hourly"
                if OLR_timefreq == "15min":
                    print("OLR is based on instantaneous 15min OLR")
                else:
                    print("OLR is based on hourly mean OLR")

                OLR_filepath = f'/data/scratch/william.keat/NEW_CP4A/{OLR_timefreq}/{dataset}/OLR/'
                if OLR_timefreq == "15min":
                    day_OLR_cube = iris.load(f'{OLR_filepath}{dataset[2:]}a.px{date2load}.pp', callback=callback_overwrite)[0][1::4]  # 15min, so every [XX:30]
                elif OLR_timefreq == "hourly":
                    day_OLR_cube = iris.load(f'{OLR_filepath}{dataset[2:]}a.pq{date2load}.pp', callback=callback_overwrite)[0]  # 15min, so every [XX:30]

                #day_OLR_cube = zoom_into_reg(day_OLR_cube, reg)

                # Get flux-equivalent brightness temperature
                day_Tf_cube = day_OLR_cube.copy()
                day_Tf_cube.data = (day_OLR_cube.data / 5.67e-8) ** (1 / 4)

                # Apply empirical adjustment, using eq1 and w1 from Ying and Slingo 2001
                day_Tb_cube = day_Tf_cube.copy()
                day_Tb_cube.data = convert_Tf_to_tb(day_Tb_cube.data)

                CL.append(day_Tb_cube)

                pr_filepath = f'/data/scratch/william.keat/NEW_CP4A/hourly/{dataset}/pr/'
                day_rain_cube = iris.load(f'{pr_filepath}{dataset[2:]}a.pq{date2load}.pp', callback=callback_overwrite)[0]
                day_rain_cube = precip_to_mm(day_rain_cube, "hour")
                pr_CL.append(day_rain_cube)

        try:
            daily_Tb_cube = CL.merge_cube()
        except:
            daily_Tb_cube = CL.concatenate_cube()
        daily_rain_cube = pr_CL.concatenate_cube()

        print(f'SHAPE Tb CUBE ={daily_Tb_cube.shape}')
        print(f'SHAPE PR CUBE ={daily_rain_cube.shape}')

        daily_rain_cube.coord("time").points = np.round(daily_rain_cube.coord("time").points, 2)  # rounding errors in time points causing time constraint problems
        daily_rain_cube.coord("latitude").guess_bounds()  # for zooming into smaller area later (generalised if image_sm>0)
        daily_rain_cube.coord("longitude").guess_bounds()

        if regrid_to_GPM_IMERG == True:
            # Need to prep the coords before regridding
            daily_Tb_cube.coord("latitude").coord_system = gpm_ref_cube.coord("latitude").coord_system
            daily_Tb_cube.coord("longitude").coord_system = gpm_ref_cube.coord("longitude").coord_system
            daily_Tb_cube.coord("latitude").guess_bounds()
            daily_Tb_cube.coord("longitude").guess_bounds()
            daily_Tb_cube = daily_Tb_cube.regrid(gpm_ref_cube, iris.analysis.AreaWeighted())  # Remap to GPM 10km grid

            daily_rain_cube.coord("latitude").coord_system = gpm_ref_cube.coord("latitude").coord_system
            daily_rain_cube.coord("longitude").coord_system = gpm_ref_cube.coord("longitude").coord_system
            daily_rain_cube = daily_rain_cube.regrid(gpm_ref_cube, iris.analysis.AreaWeighted())

        ###############################################################################################

        daily_Tb_cube = daily_Tb_cube.extract(time_con)
        daily_pr_cube = daily_rain_cube.extract(time_con)

        #############################################################################################################
        ###########################################  STARTING TRACKING   ############################################
        #############################################################################################################
        print(f'tverbose:{tverbose}')
        nt = -1  # initialied number of times into tracking. -1 so first time is 0 (used to load prev storm and opt flow).
        # Also -1 allows distinction to track 23:45 in previous

        for t in np.arange(len(daily_Tb_cube.coord("time").points)):  # This needs to be 96 (97 if all inclusive
            print(t)
            nt = nt + 1
            Tb_cube = daily_Tb_cube[t]
            pr_cube = daily_pr_cube[t]
            if nt > 0:  # This will be triggered for 2nd timestep in requested day (e.g.00:15)
                prev_Tb_cube = daily_Tb_cube[t - 1]  # previous date cube

            if use_wind_data == True:
                print("Not set up to handle model winds !!!!")
            else:
                # Use preferred optical flow method from https://gmd.copernicus.org/articles/12/1387/2019/
                if nt > 0:
                    if tverbose == True:
                        print("Calculating winds with optical flow")
                    # Winds calculated from full domain minus the 17 variable rim.
                    input_data = np.stack((prev_Tb_cube.data, Tb_cube.data), axis=0)

                    # scale input data to uint8 [0-255] with self.scaler
                    scaled_data, c1, c2 = scaler(input_data)

                    of = calculate_of(scaled_data, method=of_method, direction=direction)
                    delta_x = of[::, ::, 0]
                    delta_y = of[::, ::, 1]

                    uwind = (1000.0 * gridlength * delta_x) / (time_res_mins * 60.0)  # units of m/s
                    vwind = (1000.0 * gridlength * delta_y) / (time_res_mins * 60.0)  # units of m/s

            curr_date = Tb_cube.coord("time").units.num2date(Tb_cube.coord("time").points)[0].strftime('%Y%m%d')
            hourval = Tb_cube.coord("time").units.num2date(Tb_cube.coord("time").points)[0].hour
            minval = Tb_cube.coord("time").units.num2date(Tb_cube.coord("time").points)[0].minute

            if hourval < 10:
                hourval = '0' + str(hourval)
            else:
                hourval = str(hourval)
            if minval < 10:
                minval = '0' + str(minval)
            else:
                minval = str(minval)

            timestr = f'{curr_date}_{hourval}{minval}_UTC'
            print(timestr)
            curr_datetime = Tb_cube.coord("time").units.num2date(Tb_cube.coord("time").points)[0]  # This is used to check deltat is correct between timesteps

            if nt > 0:
                # get prev hours and mins
                prev_date = prev_Tb_cube.coord("time").units.num2date(prev_Tb_cube.coord("time").points)[0].strftime('%Y%m%d')
                prev_hourval = prev_Tb_cube.coord("time").units.num2date(prev_Tb_cube.coord("time").points)[0].hour  # .strftime('%Y%m%d_%H%M%S') # d+1 to start next day
                prev_minval = prev_Tb_cube.coord("time").units.num2date(prev_Tb_cube.coord("time").points)[0].minute

                if prev_hourval < 10:
                    prev_hourval = '0' + str(prev_hourval)
                else:
                    prev_hourval = str(prev_hourval)
                if prev_minval < 10:
                    prev_minval = '0' + str(prev_minval)
                else:
                    prev_minval = str(prev_minval)
                prev_timestr = f'{prev_date}_{prev_hourval}{prev_minval}_UTC'

                prev_datetime = prev_Tb_cube.coord("time").units.num2date(prev_Tb_cube.coord("time").points)[0]  # This is used to check deltat is correct between timesteps

                # If not the first timestep, make sure that the time difference between prev and current time correct
                if not (curr_datetime - prev_datetime).seconds == expected_delta_t:
                    raise ValueError(f'ERROR: Time difference is not {expected_delta_t}...')

            ####################################################################################################
            ####      ----------------------------  Apply storm labelling   ----------------------------    ####
            ####################################################################################################

            if verbose == True:
                print("Labelling storms")
            if sm_width_pixels > 0:
                if verbose:
                    print(f'Applying smoothing {sm_width_pixels} pixels')
                image_sm = Tb_cube.copy()
                image_sm.data = scipy.ndimage.uniform_filter(image_sm.data, size=sm_width_pixels, mode='constant')
            else:
                image_sm = Tb_cube.copy()

            # Zoom into smaller area all arrays that will be subsetted to exclude any smoothing effects.
            # This is done here BEFORE labelling. therefore boundary effects naturally not labelled.
            # Apply even if image_sm=0 for ease, otherwise different size grids.
            image_sm = zoom_into_reg(image_sm, reg_smaller).data
            Tb = zoom_into_reg(Tb_cube, reg_smaller).data  # data extracted from cube once here so not looped over to assign Tb/pr values to tracks
            pr = zoom_into_reg(pr_cube, reg_smaller).data

            curr_labels = label_objects(image_sm, min_area_pixels, threshold, struct2d)

            if solve_false_merger:
                dilated_labels = apply_erosion_dilation(curr_labels, Tb, iterations=1)  # Applied to label field, so assigning of min/max values dilates where pixel is > threshold only.
                new_Tb = np.where(dilated_labels > 0, Tb, 500)  # Tb field where dilated labels exist, 500 just arbitrary large number above 240K
                curr_labels = label_objects(new_Tb, min_area_pixels, threshold, struct2d)  # label new "original" Tb field

            A = np.histogram(curr_labels, bins=range(int(np.max(curr_labels)) + 2))  # number of pixels with each label number
            B = np.where(A[0][:] > 0)  # list of labels
            valid_labels = B[0][1:]  # skipping the first bin which is full of zeros (non-labelled area)
            numstorms = len(valid_labels)
            if verbose:
                print(f'Number of storms:{numstorms}')

            #### ----------------------------------------------------------------------------------------####
            #### ------------------------- Initialise storm property lists    -----------------------    ####
            #### ----------------------------------------------------------------------------------------####

            if numstorms > 0:
                Curr_Storms = [StormS() for i in range(numstorms)]  # WK: Initialise storm object for each storm
            else:
                Curr_Storms = []  # if no storms in current timestep, Current STorms is empty. Used at end - if empty, won't be looped over to save storm info

            if len(Prev_Storms) == 0:  # Previous Storms list initialise but contains no data - True if either first time step or no storms in previous timestep or first timestep
                if verbose:
                    print("No storms at previous timestep")
                for ns, jj in enumerate(valid_labels):
                    C = np.where(curr_labels == jj)
                    Curr_Storms[ns].storm = int(jj)
                    Curr_Storms[ns].area = int(np.size(C, 1))

                    Curr_Storms[ns].maxTb = np.max(Tb[C])
                    Curr_Storms[ns].meanTb = np.mean(Tb[C])
                    Curr_Storms[ns].minTb = np.min(Tb[C])

                    Curr_Storms[ns].maxpr = np.max(pr[C])
                    Curr_Storms[ns].meanpr = np.mean(pr[C])
                    Curr_Storms[ns].minpr = np.min(pr[C])

                    if reflectivity_weight_centroids == True:
                        Curr_Storms[ns].centroidx = np.sum(Tb[C] * xmat[C]) / np.sum(Tb[C])  # WK - reflectivity-weighted x
                        Curr_Storms[ns].centroidy = np.sum(Tb[C] * ymat[C]) / np.sum(Tb[C])  # WK - reflectivity-weighted y
                        Curr_Storms[ns].centroidlon = np.sum(Tb[C] * lonmat[C]) / np.sum(Tb[C])  # WK - reflectivity-weighted lon
                        Curr_Storms[ns].centroidlat = np.sum(Tb[C] * latmat[C]) / np.sum(Tb[C])  # WK - reflectivity-weighted lon
                    else:
                        Curr_Storms[ns].centroidx = np.mean(xmat[C])
                        Curr_Storms[ns].centroidy = np.mean(ymat[C])
                        Curr_Storms[ns].centroidlon = np.mean(lonmat[C])  # WK - added so that can plot lat/lon tracks
                        Curr_Storms[ns].centroidlat = np.mean(latmat[C])  # WK - added so that can plot lat/lon tracks

                    Curr_Storms[ns].boxleft = np.min(xmat[C])
                    Curr_Storms[ns].boxup = np.max(ymat[C])
                    Curr_Storms[ns].boxwidth = np.max(xmat[C]) - np.min(xmat[C])
                    Curr_Storms[ns].boxheight = np.max(ymat[C]) - np.min(ymat[C])

                    # C.SHORT
                    Curr_Storms[ns].was = misval
                    Curr_Storms[ns].life = 1
                    Curr_Storms[ns].track_xpos = Curr_Storms[ns].centroidx
                    Curr_Storms[ns].track_ypos = Curr_Storms[ns].centroidy
                    Curr_Storms[ns].track_lonpos = Curr_Storms[ns].centroidlon
                    Curr_Storms[ns].track_latpos = Curr_Storms[ns].centroidlat
                    Curr_Storms[ns].u = misval
                    Curr_Storms[ns].v = misval
                    Curr_Storms[ns].parent = [misval]
                    # C.SHORT Should child be a list or a single value? List here but single vlaue below....
                    Curr_Storms[ns].child = [misval]
                    Curr_Storms[ns].overlap_area_with_chosen_advected_storm = [misval]
                    Curr_Storms[ns].accreted = [misval]

                    Curr_Storms[ns].primary_tracked = False
                    Curr_Storms[ns].secondary_tracked = False
                    Curr_Storms[ns].initiated = True  # All storms set initiated to T on first timestep
                    Curr_Storms[ns].split = False  # Other properties/flags set to False
                    Curr_Storms[ns].merged = False
                    Curr_Storms[ns].reached_max_dist = False

                    Curr_Storms[ns].x = xmat[C]
                    Curr_Storms[ns].y = ymat[C]
                    Curr_Storms[ns].lons = lonmat[C]
                    Curr_Storms[ns].lats = latmat[C]
                    Curr_Storms[ns].Tb = Tb[C]  # Added so that can look at Tb of individual objects
                    Curr_Storms[ns].pr = pr[C]  # Added so that can easily use precip to filter/define MCSs

            ####      ----------------------------------------------------------------------------------------####
            elif np.max(prev_labels) > 0 and np.max(curr_labels) > 0:  # WK: Else, if there are storms in pervious timestep, and storms are found in current time step
                if verbose:
                    print("Storms present at previous and current timestep")
                ####  -------------------------------   ADVECT PREVIOUS STORM LABELS-----------------------------####
                if tverbose:
                    print("Advecting previous labels")
                advected_labels, advected_Tb, advected_labels_clash = advect_previous_labels(prev_labels, Prev_Storms, prev_Tb, uwind, vwind, time_res_mins, gridlength, dynamic_maxdist, verbose=False)
                if verbose:
                    print(f'ADVECTED LABEL CLASH COUNTER:{advected_label_clash_counter}')
                ####      ----------------------------------------------------------------------------------------####

                # SAVE ADVECTED LABEL POSITIONS AND PROPERTIES FOR MERGER/SPLIT LOGIC LATER
                # Loop through advected storms (same indices as Pre Storms) --> save the centre indcs and areas of advected labels
                # This is then used later if there arare multiple overlaps to determine which is largest and closest
                if tverbose:
                    print("Loop over advected storms. Used to check for overlaps later")
                for ns in range(len(Prev_Storms)):
                    jj = Prev_Storms[ns].storm  # Indices of previous storm.
                    advectedind = np.where(advected_labels == jj)  # indices of advected storm
                    if np.size(advectedind, 1) == 0:  # If no advected indices (presumably if advected outside of domain?)
                        if verbose:
                            print(f'Advected storm {jj} has no area, presumably advected outside of the domain?')
                        continue
                    else:
                        if reflectivity_weight_centroids == True:  # WK - to impement reflectivity-weighting, added functionality to produce "advected Tb" field in advect_labels(). Needed for centroid separation interaction logic...
                            Advected_Storm_Properties[ns][0] = np.sum(advected_Tb[advectedind] * xmat[advectedind]) / np.sum(advected_Tb[advectedind])  # Mean x of storm. (confirms old logic on y<-->x, u<-->v indexing when assigning newlabels above. centrid is [y,x], so when subbing xmat does it correctly.
                            Advected_Storm_Properties[ns][1] = np.sum(advected_Tb[advectedind] * ymat[advectedind]) / np.sum(advected_Tb[advectedind])  # Mean y of storm
                            Advected_Storm_Properties[ns][2] = int(np.size(advectedind, 1))  # Number of pixels where overlap
                        else:
                            Advected_Storm_Properties[ns][0] = np.mean(xmat[advectedind])  # Mean x of storm. (confirms old logic on y<-->x, u<-->v indexing when assigning newlabels above. centrid is [y,x], so when subbing xmat does it correctly.
                            Advected_Storm_Properties[ns][1] = np.mean(ymat[advectedind])  # Mean y of storm
                            Advected_Storm_Properties[ns][2] = int(np.size(advectedind, 1))  # Number of pixels where overlap

                ###############################################################################
                # NOW LOOP THROUGH CURRENT STORMS AND CHECK FOR OVERLAP WITH ADVECTED STORMS  #
                ###############################################################################

                # Thibit calculates advected_storm_areas,
                wasnum = np.zeros(len(Curr_Storms))  # initiliase array to store labels of what current storms used to be
                overlap_bins = np.arange(0, np.max(prev_labels) + 2)  # Needs to have 2 more than # prev labels for histogram bins
                advected_storm_areas = np.ones([int(np.max(prev_labels)) + 1])
                for qq in range(len(Prev_Storms)):  # Loop over previous storms
                    if Advected_Storm_Properties[qq, 2] > 0:  # If area of advected storm > 0 (this wouldn't be the case if storm advected out of domain)
                        # C.SHORT This means that a storm with zero area (Advected_Storm_Properties[qq, 2] = 0) would have a value of 1 in advected_storm_areas QUESTION
                        advected_storm_areas[qq + 1] = Advected_Storm_Properties[qq, 2]  # qq+1 as first bin is just everything that is not an object
                    else:
                        if verbose:
                            print(f'(advected storm area is 0 because storm {qq} advected outside of the domain')

                # Definitely duplication here - could add the "if Advected_Storm_Properties" bit to loop where Advected_Storm_Properties defubed
                # Not sure what the big idea is with the +1... also appears later with qindex+1

                if tverbose:
                    print("(Major section): Loop through current storms and check for overlaps")

                numstorms = len(valid_labels)
                for ns, jj in enumerate(valid_labels):  # Loop through all current storms and give them their properties
                    if verbose:
                        print(f'############################   CURRENT STORM (ns={ns},jj={jj})) ##########################')
                    C = np.where(curr_labels == jj)  # C = indices of storm with current labels
                    Curr_Storms[ns].storm = int(jj)
                    Curr_Storms[ns].area = int(np.size(C, 1))

                    Curr_Storms[ns].maxTb = np.max(Tb[C])
                    Curr_Storms[ns].meanTb = np.mean(Tb[C])
                    Curr_Storms[ns].minTb = np.min(Tb[C])

                    Curr_Storms[ns].maxpr = np.max(pr[C])
                    Curr_Storms[ns].meanpr = np.mean(pr[C])
                    Curr_Storms[ns].minpr = np.min(pr[C])

                    if reflectivity_weight_centroids == True:
                        Curr_Storms[ns].centroidx = np.sum(Tb[C] * xmat[C]) / np.sum(Tb[C])  # WK - reflectivity-weighted x
                        Curr_Storms[ns].centroidy = np.sum(Tb[C] * ymat[C]) / np.sum(Tb[C])  # WK - reflectivity-weighted y
                        Curr_Storms[ns].centroidlon = np.sum(Tb[C] * lonmat[C]) / np.sum(Tb[C])  # WK - reflectivity-weighted lon
                        Curr_Storms[ns].centroidlat = np.sum(Tb[C] * latmat[C]) / np.sum(Tb[C])  # WK - reflectivity-weighted lon
                    else:
                        Curr_Storms[ns].centroidx = np.mean(xmat[C])
                        Curr_Storms[ns].centroidy = np.mean(ymat[C])
                        Curr_Storms[ns].centroidlon = np.mean(lonmat[C])  # WK - added so that can plot lat/lon tracks
                        Curr_Storms[ns].centroidlat = np.mean(latmat[C])  # WK - added so that can plot lat/lon tracks

                    # C.SHORT
                    #Curr_Storms[ns].was = int(jj)  # Start each s torms "was" label as it's current.
                    Curr_Storms[ns].was = misval
                    Curr_Storms[ns].life = 1  # Always set to 1, will be updted with Prev Storm label if overlap found
                    Curr_Storms[ns].track_xpos = Curr_Storms[ns].centroidx
                    Curr_Storms[ns].track_ypos = Curr_Storms[ns].centroidy
                    Curr_Storms[ns].track_lonpos = Curr_Storms[ns].centroidlon
                    Curr_Storms[ns].track_latpos = Curr_Storms[ns].centroidlat
                    # C.SHORT
                    #Curr_Storms[ns].u = np.mean(uwind[C])
                    #Curr_Storms[ns].v = np.mean(vwind[C])
                    Curr_Storms[ns].u = misval
                    Curr_Storms[ns].v = misval
                    Curr_Storms[ns].parent = [misval]
                    Curr_Storms[ns].child = [misval]
                    Curr_Storms[ns].overlap_area_with_chosen_advected_storm = [misval]
                    Curr_Storms[ns].accreted = [misval]

                    # Need to check whether these need to be re-initialised
                    Curr_Storms[ns].primary_tracked = False
                    Curr_Storms[ns].secondary_tracked = False
                    Curr_Storms[ns].initiated = False
                    Curr_Storms[ns].split = False
                    Curr_Storms[ns].merged = False
                    Curr_Storms[ns].reached_max_dist = False

                    Curr_Storms[ns].x = xmat[C]
                    Curr_Storms[ns].y = ymat[C]
                    Curr_Storms[ns].lons = lonmat[C]
                    Curr_Storms[ns].lats = latmat[C]
                    Curr_Storms[ns].Tb = Tb[C]
                    Curr_Storms[ns].pr = pr[C]  # Added so that can easily use precip to filter/define MCSs

                    ###################################################
                    # CHECK OVERLAP WITH QHIST
                    # IF NO OVERLAP, THEN
                    # GENERATE (halo) km RADIUS AROUND CENTROID
                    # CHECK FOR OVERLAP WITHIN (halo) km OF CENTROID
                    ###################################################

                    # Histogram of number of pixels that each advected storm overlaps with current storm jj
                    # First histogram is normalised by current storm area (to produce fractional overlap)
                    overlap_bins = range(int(np.max(prev_labels)) + 2)  # Needs to have 2 more than # prev labels for histogram bins
                    # ^^^Any positive overlap values now only correspond to real storm labels.

                    # qhist1 referes to fractional overlap of advected storms with current storm with respect to current storm area
                    # qhist2 refers to fractional overlap of advected storms with current storm but with respect to the advected storm area
                    # Following TITAN methodology, the sum of these is taken before comparing with threshold of 0.6
                    # Perfect overlap would make each component=1, giving max value of 2.
                    qhist1 = (np.histogram(advected_labels[np.where(curr_labels == jj)], overlap_bins))[0][:] / float(Curr_Storms[ns].area)
                    qhist2 = (np.histogram(advected_labels[np.where(curr_labels == jj)], overlap_bins))[0][:] / advected_storm_areas[:]
                    overlap_hist = qhist1 + qhist2  # ignore first bin as that is all the non-labelled bits (zeroes)

                    #If overlap threshold is less than threshold, add a halo and try again
                    # NOTE - the addition of a halo around current storm without modifying current storm area will change meaning of overlap_hist - 2 no longer upper limit
                    # However, hereafter overlap_hist values are used for relative purposes(e.g. to pick biggest overlaps) so doesn't matter.
                    # GENERATE (halo) km RADIUS AROUND CENTROID #
                    if np.max(overlap_hist[1:]) < lapthresh:
                        if verbose:
                            print(f'No advected label overlaps with current storm label {jj}. Adding halo to current storm')

                        # By adding a halo/blob we have effectively modified the area of the current storm. Now use this new area in the
                        newblob = 0 * xmat
                        blobind = np.where((xmat - Curr_Storms[ns].centroidx) ** 2 + (ymat - Curr_Storms[ns].centroidy) ** 2 < halosq)
                        newblob[blobind] = newblob[blobind] + 1  # circular blob of ones about centroid

                        # overlap calculation
                        blob_labels = 0 * xmat
                        blob_labels[newblob == 1] = jj
                        newblob_area = np.count_nonzero(newblob)
                        qhist1 = (np.histogram(advected_labels[np.where(newblob == 1)], overlap_bins))[0][:] / float(newblob_area)
                        qhist2 = (np.histogram(advected_labels[np.where(newblob == 1)], overlap_bins))[0][:] / advected_storm_areas[:]
                        overlap_hist = qhist1 + qhist2

                        if np.max(overlap_hist[1:]) >= lapthresh:
                            Curr_Storms[ns].secondary_tracked = True  # counting number of times "secondary" tracking method used (halo used)
                    else:
                        Curr_Storms[ns].primary_tracked = True  # counting number of times "primary" tracking successful (overlap>-0.6 without need for halo)

                    if verbose:
                        print(f'overlap_hist[1:]= {overlap_hist[1:]}')
                    #############################################################################################################################
                    # IF OVERLAP, THEN INHERIT "WAS", UPDATE "LIFE" AND "TRACK" AND "WASDIST", INHERIT "U" AND "V" (ONLY UPDATE IF SINGLE OVERLAP)
                    ############################################################################################################################

                    if np.max(overlap_hist[1:]) >= lapthresh:  # if advected label overlap wrt current storm + wrt itself >0.6, assign current label to advected and update cell life (+1)

                        labels_that_overlap = np.where(overlap_hist[1:] >= lapthresh)  # skipping bin 0 as that is all non-labelled areas
                        # When looping over current storms, ns =0 corresponds to valid label 1, so 1: indexing below works as ignores invalid labe (0)
                        ###################################################
                        # IF MORE THAN ONE GOOD OVERLAP, KEEP PROPERTIES OF STORM WITH LARGEST OVERLAP
                        # IF MORE THAN ONE LARGEST, KEEP NEAREST IN CENTROID
                        ###################################################

                        if np.size(labels_that_overlap, 1) > 1:  # If more than one overlap
                            if verbose:
                                print('###########   More than one good overlap found, looping over each found overlap  ##########')
                            lapdist = np.zeros([np.size(labels_that_overlap, 1)])  # initialise array to store curr vs advected centroid distances
                            sectlap = np.zeros([np.size(labels_that_overlap, 1)])  # initialise array to store areas of advected and current overlap
                            for overlap_label_idx in range(np.size(labels_that_overlap, 1)):  # Loop over each overlap, checking against pre-calculated Advected_Storm_Properties above test for largest
                                chosen_overlap_label = np.squeeze(labels_that_overlap[0][overlap_label_idx])
                                if verbose:
                                    print(f'chosen_overlap_label {chosen_overlap_label}')
                                # The centre of storm overlap areas between current storm jj and advected labels used for distance calcs below
                                lapdist[overlap_label_idx] = np.sqrt((Curr_Storms[ns].centroidx - Advected_Storm_Properties[chosen_overlap_label, 0]) ** 2 + (Curr_Storms[ns].centroidy - Advected_Storm_Properties[chosen_overlap_label, 1]) ** 2)
                                # C.SHORT Why do we need to add 1 to chosen_overlap_label? QUESTION
                                sectlap[overlap_label_idx] = np.size(np.where((advected_labels == chosen_overlap_label + 1) & (curr_labels == jj)), 1)  # area of overlap for each of the more than 1 found
                                # The 'chosen_overlap_label + 1' because idx is from overlap_hist[1:] and labelling starts from 0
                                # Advected_Storm_Properties (list of objects) only exist for valid storms by definition, so index is correct (already+1)
                            idcs_max_overlap = np.where(sectlap == np.max(sectlap))  # list of indices where the overlap is the same as the max. If more than one, want to keep LARGEST OVERLAP
                            if np.size(idcs_max_overlap, 1) > 1:  # If more than one advected storm overlaps with the current storm by the same amount (max value)
                                if verbose:
                                    print(f'{np.size(idcs_max_overlap,1)} storms overlap current storm by same amount. Choosing advected storms CLOSEST to current storm')
                                closest_storm = idcs_max_overlap[0][np.where(lapdist[idcs_max_overlap[0][:]] == np.min(lapdist[idcs_max_overlap[0][:]]))]  # indices of closest
                                if len(closest_storm) > 1:
                                    print(f'{len(closest_storm)} storms the same distance away. Just picking first in list (insufficient info to distinguish which storm better)')
                                closest_storm = closest_storm[0]  # Either there is one closest storm (so this is correct) or more than one. In rare case, still just pick [0] as no extra info
                                if verbose:
                                    print(f'Just picked first storm ({closest_storm}) from list (not sufficient info to distinguish which storm better)')
                            else:  # simple case- there is only one storm with maximum overlap - just pick it.
                                closest_storm = idcs_max_overlap[0][0]  # Closest storm is actually just the storm with with largest overlap (of which there is only one to trigger this else statement)
                                if verbose:
                                    print(f'Single storm with maximum overlap with curr storm {jj}: prev label {closest_storm}')

                            chosen_overlap_label = np.squeeze(labels_that_overlap[0][closest_storm])
                            if verbose:
                                print(f'###=========CHOSEN OVERLAP LABEL FOR STORM jj:{jj} is prev label:{chosen_overlap_label} ({timestr})=======#####')

                            Curr_Storms = assign_historical_tracks_to_current_storm(Curr_Storms, Prev_Storms, ns, chosen_overlap_label)

                            # All other storms that were not chosen but that overlap enough > lapthresh are all merged (added to accreted list_
                            if verbose:
                                print("DEALING WITH MERGERS")

                            alllaps = np.where(overlap_hist[1:] >= lapthresh)  # all overlaps, 1: to ignore bin 0 which is index of non-storms
                            for overlap_label_idx in range(np.size(alllaps, 1)):  # loop over each label index (prev labels that overlap)
                                if verbose:
                                    print(f'overlap label idx {overlap_label_idx} ')
                                allindex = np.squeeze(alllaps[0][overlap_label_idx])
                                if allindex == chosen_overlap_label:  # If considering labels of storm that has been chosen as correct prev storm
                                    continue  # move on to consider next overlap as this label is already used as correct previous label
                                # Below if/else just to handle assignment depending on whether this is first accretion or not
                                if Curr_Storms[ns].accreted[-1] == misval:  # If current storm not accreted any storms yet, add single value
                                    # C.SHORT
                                    #Curr_Storms[ns].accreted[-1] = Prev_Storms[allindex].was # all of the overk
                                    Curr_Storms[ns].accreted[-1] = Prev_Storms[allindex].storm
                                    Curr_Storms[ns].merged = True  # Flag for ease to se if merged
                                else:  # If current storm already has accreted a storm, add this one to that list
                                    # C.SHORT
                                    #Curr_Storms[ns].accreted.append(Prev_Storms[allindex].was)
                                    Curr_Storms[ns].accreted.append(Prev_Storms[allindex].storm)
                                    Curr_Storms[ns].merged = True
                                if verbose:
                                    print(f'Current storm label {jj} accretes previous storm label {overlap_label_idx}')

                            # Calculate overlap area of chosen advected storm with current storm (could also be done by indexing qhist again?)
                            Curr_Storms[ns].overlap_area_with_chosen_advected_storm = np.size(np.where((advected_labels == chosen_overlap_label + 1) & (curr_labels == jj)), 1)  # chosen_overlap label +1 due to indexing

                        ###################################################
                        # SINGLE OVERLAP
                        ###################################################
                        else:
                            chosen_overlap_label = np.squeeze(labels_that_overlap[0][0])
                            if verbose:
                                print(f'###########  Single overlap found with storm {jj}: prev label ={chosen_overlap_label} ##########')
                            # Assign previous tracks for this label to current storm
                            if verbose:
                                print(f'###=========CHOSEN OVERLAP LABEL FOR STORM jj:{jj} is prev label:{chosen_overlap_label} ({timestr})=======#####')

                            Curr_Storms = assign_historical_tracks_to_current_storm(Curr_Storms, Prev_Storms, ns, chosen_overlap_label)
                            # Calculate the overlap between chosen overlap storm and current labels (overlap_area_with_chosen_advected_storm used to be overlap_area_with_chosen_advected_storm, but not clear at all what it is)
                            # If halo was used (secondary_tracked=T) the area of the current cell was effectively modified. Make sure to use this new area
                            # when computing overlap
                            if Curr_Storms[ns].primary_tracked:
                                Curr_Storms[ns].overlap_area_with_chosen_advected_storm = np.size(np.where((advected_labels == chosen_overlap_label + 1) & (curr_labels == jj)), 1)
                            elif Curr_Storms[ns].secondary_tracked:
                                Curr_Storms[ns].overlap_area_with_chosen_advected_storm = np.size(np.where((advected_labels == chosen_overlap_label + 1) & (blob_labels == jj)), 1)
                            #TODO Some of the above is duplicated. Just need to assign chosen_overlap_label. Curr_Storms[ns], assigning tracks not needed in each if statement

                    ###################################################
                    # IF NO OVERLAP, THEN NEW STORM
                    # - "WAS" SET TO CURRENT MAX LABEL +1  presumably to ensure unique)
                    # - UPDATE "LIFE" AND "TRACK" AND "WASDIST" FOR A NEW STORM
                    ###################################################
                    else:
                        Curr_Storms[ns].initiated = True
                        # C.SHORT
                        # Curr_Storms[ns].was = safe_was_label # Needs a was label for next time step
                        Curr_Storms[ns].was = misval
                        Curr_Storms[ns].life = 1
                        # C.SHORT
                        # safe_was_label = safe_was_label + 1

                    wasnum[ns] = Curr_Storms[ns].was  # array of what historical storm labels for each current storm

                ####################################################################################################################################
                # MAKE SURE THAT IF ADVECTED OVERLAP IS CANDIDATE FOR ACCRETION BUT ALSO IS PRIMARY TRACK, PRIMARY TRACK PRIORITISED OVER ACCRETION
                ####################################################################################################################################
                for ns in range(len(Curr_Storms)):
                    jj = Curr_Storms[ns].storm
                    if Curr_Storms[ns].accreted[-1] == misval:  # if not accreted anything, next storm
                        continue  # next storm
                    else:  # Current storm has accreted at least on storm (at least as prev accreted storms could themsleves have accreted others)
                        if verbose:
                            print("ACCRETED STORMS BEFORE AC CODE")
                            print(Curr_Storms[ns].accreted)
                            print("")
                        for acnum in range(np.size(Curr_Storms[ns].accreted)):  # loop over all the accreted storm label;s
                            # C.SHORT Don't want to consider storms with no accreted storms
                            if Curr_Storms[ns].accreted[acnum] == misval:
                                print(Curr_Storms[ns].accreted)
                                raise ValueError("HI")
                                continue
                            acind = np.where((wasnum - Curr_Storms[ns].accreted[acnum]) == 0)  # Are any accreted storms (of jj) in any of the was labels of current storms (wasnum)
                            if np.size(acind, 1) > 0:  # If this is the case

                                # C.SHORT
                                if Curr_Storms[ns].accreted[acnum] == misval:
                                    if verbose:
                                        print(acind)
                                        print(wasnum[acind])
                                    raise ValueError("UH-OH")

                                Curr_Storms[ns].accreted[acnum] = misval  # remove it from the accreted list
                        acnew = [aci for aci in Curr_Storms[ns].accreted if aci > misval]  # creates new list of accreted labels with these missing
                        if verbose:
                            print(f'CURR STORMS ACCRETED VALS = {Curr_Storms[ns].accreted}')

                        if np.size(acnew) > 0:
                            Curr_Storms[ns].accreted = acnew  # WK - just adding list rather than needless looping (and acnum error) in original code
                        else:
                            Curr_Storms[ns].accreted = [misval]
                        if verbose:
                            print("ACCRETED STORMS AFTER AC CODE:")
                            print(Curr_Storms[ns].accreted)
                            print("")

                if tverbose:
                    print("Primary and secondary Tracks identified")

                ###################################################
                # TRACKING MERGING BREAKING
                # MULTIPLE STORMS AT T (M) MAY HAVE SAME LABEL "WAS"
                # FIND STORM WITH LARGEST OVERLAP AT T+1 WITH ADVECTED q(T)
                # THIS IS THE "PARENT" STORM,
                # "PARENT" VECTOR WITH INDICES OF NEW LABELS FOR "CHILD" STORMS
                # STORMS WITH SAME WAS BUT FURTHER FROM CENTROID ARE "CHILD", VALUE "PARENT"
                ###################################################

                if tverbose:
                    print("Dealing with mergers/splits")
                # All below does is create an array filled with the area overlap of each storm with same "was" label as the current storm's "was" label
                for ns in range(len(Curr_Storms)):  # Loop over all current storms # TODO Does this loop need to be separate? Whole chunk I think could be be simplified
                    jj = Curr_Storms[ns].storm  # jj = current storm label
                    if Curr_Storms[ns].overlap_area_with_chosen_advected_storm == [misval]:  # if overlap between chosen storm and current storm is misval, should not be considered (does not overlap)
                        continue  # As Curr_Storms[ns].overlap_area_with_chosen_advected_storm initalised with misval, skips new storms (only considers those with chosen prev label....

                    # C.SHORT Don't want to consider storms that didn't exist at previous timestep/ WK: This step is not in Stein version... Not sure effect of adding/removing it?
                    if wasnum[ns] == misval:
                        continue

                    # Need to see if more than one current storm has the same previous label (i.e was label).
                    same_was_ind = np.where(wasnum == wasnum[ns])  # current storm indices with same "was" label as current storm. Multiple means more than 1 current storm has same "was" label [SPLIT].
                    overlap_was_array_length = 0  # Just a counter to define length of array later - #TODO suspect this can be simplified... use list?
                    # for loop below is purely for counting to initialised overlap was array below
                    for storm_was_idx in range(np.size(same_was_ind)):  # loop over each Current Storm that has the same "was" label as Current Storm[ns]
                        if Curr_Storms[same_was_ind[0][storm_was_idx]].overlap_area_with_chosen_advected_storm == [misval]:  # As above, but now checking if storm with the same "was" label overlaps with current storm ns
                            # C.SHORT How can this ever be triggered? If a storm has a was label, how can it not overlap with anything?
                            continue  # skip if storm with same "was" labels does not overlap (hence is equal to initialised value (misval))
                        else:
                            overlap_was_array_length = overlap_was_array_length + 1  # add one to length for array defining below

                    # Now, initialised overlap_was_array and go back over these storms to assign overlap areas of storms with the same "was" label as current storm
                    overlap_was_array = np.zeros(overlap_was_array_length)  # initialise array to fill with each overlap
                    if np.size(same_was_ind) > 1:  # If more than one storm has same "was" label as current storm
                        kkval = 0  # new counter for assigning overlap areas to overlap_was_array
                        for storm_was_idx in range(np.size(same_was_ind)):  # loop over each Current Storm that has the same "was" label as Current Storm[ns]
                            if Curr_Storms[same_was_ind[0][storm_was_idx]].overlap_area_with_chosen_advected_storm == [misval]:  # Same condition as above: checking if storm with the same "was" label overlaps with current storm ns
                                continue  # skip if storm with same "was" labels does not overlap (hence is equal to initialised value (misval))
                            else:  # it is not a missing value which means that it overlaps by whatever the value is.
                                overlap_was_array[kkval] = Curr_Storms[same_was_ind[0][storm_was_idx]].overlap_area_with_chosen_advected_storm
                                kkval = kkval + 1  # counter for assigning to array
                    else:
                        overlap_was_array = Curr_Storms[same_was_ind[0][0]].overlap_area_with_chosen_advected_storm  # simpler case as only one storm has same "was" label.

                    ##############################################################################
                    # OVERLAP WAS ARRAY NOW CONTAINS ALL NON-ZERO OVERLAP VALUES WITH CURRENT STORM
                    # FIND THE MAXIMUM (THIS WILL BE THE PARENT) ALL OTHER STORMS WILL BE THE CHILDREN
                    ###############################################################################
                    if verbose:
                        print(f'OVERLAP WAS ARRAY = {overlap_was_array}')
                    kmax = np.where(overlap_was_array == np.max(overlap_was_array))  # index in overlap_was_array where biggest overlap of all "was" storms with current storm
                    kkmax = np.min(kmax)  # index of min, just first index if more than one I think
                    if verbose:
                        print(f'overlap_was_array = {overlap_was_array}')
                        print(f'kmax = {kmax}')
                        print(f'kkmax = {kkmax}')
                    # !!! #TODO Looks as if when multiple storms have same was label and same overlap, just picks first index. Can this be improved?

                    # Below, loop through and sort out sort
                    children = []
                    for storm_was_idx in range(np.size(overlap_was_array)):  # for each storm
                        if not storm_was_idx == kkmax:  # if not the maximum overlap index, then not the parent:
                            Curr_Storms[same_was_ind[0][storm_was_idx]].split = True  # These storms have split BUT ARE NOT THE PARENT
                            # C.SHORT.      WK: this bit makes sense, but is still not in the latest "Simple Track" version.
                            if Curr_Storms[same_was_ind[0][storm_was_idx]].child == [misval]:  # misval is by default
                                Curr_Storms[same_was_ind[0][storm_was_idx]].child = [Curr_Storms[same_was_ind[0][kkmax]].storm]  # Take label of parent storm
                            elif Curr_Storms[same_was_ind[0][kkmax]].storm not in Curr_Storms[same_was_ind[0][storm_was_idx]].child:
                                Curr_Storms[same_was_ind[0][storm_was_idx]].child.append(Curr_Storms[same_was_ind[0][kkmax]].storm)
                            else:
                                pass

                            Curr_Storms[same_was_ind[0][storm_was_idx]].was = Curr_Storms[same_was_ind[0][kkmax]].was  #  Make was a new label that is safe (i.e uniquely different)
                            #Curr_Storms[same_was_ind[0][storm_was_idx]].child=Curr_Storms[same_was_ind[0][kkmax]].was # Take label of previous storm
                            #Curr_Storms[same_was_ind[0][storm_was_idx]].was=safe_was_label #  Make was a new label that is safe (i.e uniquely different)
                            #Curr_Storms[same_was_ind[0][storm_was_idx]].life=1 # Rset to 1 for spawned "Child" cls As opposed tol parent life before: Curr_Storms[same_was_ind[0][kkmax]].life
                            Curr_Storms[same_was_ind[0][storm_was_idx]].life = Curr_Storms[same_was_ind[0][kkmax]].life  #Take on life of the storm wih biggest overlap

                            wasnum[same_was_ind[0][storm_was_idx]] = Curr_Storms[same_was_ind[0][storm_was_idx]].was
                            # C.SHORT
                            # children.append(Curr_Storms[same_was_ind[0][storm_was_idx]].was)
                            children.append(Curr_Storms[same_was_ind[0][storm_was_idx]].storm)
                            Curr_Storms[same_was_ind[0][storm_was_idx]].overlap_area_with_chosen_advected_storm = misval

                            #safe_was_label=safe_was_label+1 # add 1 to safe label to ensure new,

                    ###################################################
                    # UPDATE PARENT STORM WITH CHILDREN
                    ###################################################
                    # THIS BIT DEALS ONLY WITH KKMAX (i.e the parent, not dealt with in above loop)
                    if np.size(children) > 0:
                        Curr_Storms[same_was_ind[0][kkmax]].parent = children  # it is the parent to the children list created above

            ###################################################
            # TRACKING AND WRITING DONE SO PREPARE SPACE
            # FOR NEXT TIME STEP
            ###################################################

            # C.SHORT
            # Take a copy of Prev_Storms, set storm advection velocity to correct value, then use
            # output Prev_Storms_copy to overwrite pickle file of storms output at previous
            # timestep (see below)
            Prev_Storms_Copy = Prev_Storms.copy()
            for ns in range(len(Prev_Storms_Copy)):  # Loop over each storm from previous time step
                jj = Prev_Storms_Copy[ns].storm  # previous storm label
                labelind = np.where(prev_labels == jj)  # Indices of previous storm jj
                Prev_Storms_Copy[ns].u = np.mean(uwind[labelind])  # Mean of all u shifts for that storm (in pixels)
                Prev_Storms_Copy[ns].v = np.mean(vwind[labelind])  #

            if np.max(curr_labels) > 0:  # If storm labels exist, save current Tb as qTb to be used as prev Tb in next time step
                # For making regression line for overlap checking

                Prev_Storms = Curr_Storms  # WK - last time step objects set to current
                prev_labels = curr_labels  # WK - last time step labels set to current
                prev_Tb = Tb + 0.0  # WK - last time step Tb set to current TODO - prev Tb was for autocorrelation u and v estimation so would only need to keep if option
                Advected_Storm_Properties = np.zeros([numstorms, 3])  # For saving centroidx, y and areas of advected storms.
                ## numstorms in np.zeros() above is for current ts --> will be correct number for next ts prev labels
            else:
                # WK: if no storm labels, reset
                if verbose:
                    print('No storms detected in this timestep, resetting Prev_Storms = []')
                Prev_Storms = []

            # Save storm data
            print("AREAS", [Curr_Storms[i].area for i in np.arange(len(Curr_Storms))])
            pickle.dump(Curr_Storms, open(f'{OUT_DIR}{timestr}_storms.p', "wb"))
            if nt > 0:
                pickle.dump(Prev_Storms_Copy, open(f'{OUT_DIR}{prev_timestr}_storms.p', "wb"))
                print(f'{timestr}: Saved storm pickle to: {OUT_DIR}')
                sys.stdout.flush()
            # Save advected labels array
            try:
                pickle.dump(advected_labels, open(f'{OUT_DIR}{timestr}_advected_labels.p', "wb"))
            except:
                print("advected_labels array does not exist yet")
                pass

        print(f'TRACKING COMPLETED FOR DAY {date2load} of {full_date_range[0]}-{full_date_range[-1]})')
        sys.stdout.flush()

    timestamp_end = dt.datetime.now()
    print(f'****** @  {timestamp_end}: Finished! **************')
    timediff_total = timestamp_end - timestamp_start
    print(f'########## Total run time:  {timediff_total} \n')

    print("===========================================================")
    print("           ALL DONE!")
    print("===========================================================")