import iris
import xarray as xr
import pandas as pd
import glob
import os
import datetime
import glob
import argparse
import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.quickplot as qplt
import numpy as np
import cartopy.crs as ccrs
import scipy.stats as stats
import cartopy.geodesic as cgeo
import warnings
from scipy.ndimage import label
import time
warnings.filterwarnings("ignore")

### READ BEFORE USE ###
# Certain aspects of this code REQUIRE CUSTOMISATION to suit user data.
# To find these, search for TAG string.
# Critical note - the code assumes tracking on a lat/lon ordered grid. To use this, you needed to transpose PRIOR TO TRACKING if lon/lat.



# TAG TAG TAG
################################################################################
#### !!! USER CUSTOMISATION REQUIRED !!! ###

# Set paths and parameters
raw_track_dir="/gws/nopw/j04/kscale/USERS/bmaybee/simpleTrack_MCS_outputs/" # location of raw simpleTrack .txt outputs
raw_track_str="history_S50_T241_A9" # .txt file suffix (up to date info)
output_dir=raw_track_dir # output directory for compiled tables and MCS mask .nc files
period=pd.date_range("2016-08-01 00:00","2016-09-09 23:00",freq="h") # Period covered by tracking
deltaX,deltaY=11.1,11.1 # longitudinal and latitudinal grid spacing on which tracking was done (km).
xcoord,ycoord="longitude","latitude"

# Different data sources have different formats - this needs customising to MATCH YOUR DATA/DOMAIN. 
# This code is for the post-processed DYAMOND2 kscale hierarchy data available on jasmin (/gws/nopw/j04/kscale/DATA/)
def load_file(ffile,var):
    if var=="all":
        ds=xr.open_dataset(ffile)
    else:
        ds=xr.open_dataset(ffile)[var]
        
    try:
        test=ds.latitude.min()
    except:
        ds=ds.rename({"lat":"latitude","lon":"longitude"})
        
    if region == "sahel":
        ds=ds.sel(latitude=slice(9,19),longitude=slice(-12,12))
    if region == "wafrica":
        ds=ds.sel(latitude=slice(-6,24),longitude=slice(-18,32))
    if region == "ea-waf":
        ds=ds.sel(latitude=slice(0,25),longitude=slice(-40,30))
    if region == "safrica":
        ds=ds.sel(latitude=slice(-35,-15),longitude=slice(20,35))
    if region == "samerica":
        #Note not full box we want, due to lam constraint; ideally latitude=slice(-40,-20)
        ds=ds.sel(latitude=slice(-40,-20.5),longitude=slice(-68,-47))
    if region == "india":
        #Note not full box we want, due to channel constraint; ideally latitude=slice(5,30)
        ds=ds.sel(latitude=slice(5,30),longitude=slice(70,90))
    if region == "aus":
        #Note not full box we want, due to lam constraint; ideally latitude=slice(-23,-11)
        ds=ds.sel(latitude=slice(-23,-11),longitude=slice(120,140))
    if region == "summer" or region=="winter:
        ds=ds.sel(latitude=slice(-35,-25))
    return ds

################################################################################

def olr_to_bt(olr):
    #Application of Stefan-Boltzmann law
    sigma = 5.670373e-8
    tf = (olr/sigma)**0.25
    a = 1.228
    b = -1.106e-3
    Tb = (-a + np.sqrt(a**2 + 4*b*tf))/(2*b)
    return Tb

def get_track_info(dict,sim,tstamp):    
    with open(raw_track_dir+f"{raw_track_str}_{tstamp}02.txt","r") as data_file: # NOTE - MAY NEED TO CHANGE FINAL 02 MINUTE SUFFIX
        for line in data_file:
            data = line.split()
            if len(data) > 5:
                storm_id=data[1]
                if storm_id in dict.keys():
                    pass
                else:
                    dict[storm_id]=[storm_id,tstamp]
                dict[storm_id].append(deltaX*deltaY*float([d for d in data if d.startswith('area=')][0].replace('area=','')))
                dict[storm_id].append([d for d in data if d.startswith('box=')][0].replace('box=',''))
                dict[storm_id].append(ref_lons[int(float([d for d in data if d.startswith('centroid=')][0].replace('centroid=','').split(',')[0]))])
                dict[storm_id].append(ref_lats[int(float([d for d in data if d.startswith('centroid=')][0].replace('centroid=','').split(',')[1]))])
                #Multiplication factor converts rawstorm speeds from pixels/timestep -> km/h:
                dict[storm_id].append(deltaX*float([d for d in data if d.startswith('dx=')][0].replace('dx=','')))
                dict[storm_id].append(deltaY*float([d for d in data if d.startswith('dy=')][0].replace('dy=','')))
                dict[storm_id].append(float([d for d in data if d.startswith('extreme=')][0].replace('extreme=','')))
                dict[storm_id].append(float([d for d in data if d.startswith('meanv=')][0].replace('meanv=','')))
    return dict

#### CRUCIAL NOTE: THIS BUILDS TABLE FOR DATA ASSUMING LAT/LON GRID! 
# Otherwise column labels indicate wrong direction
def build_table(sim):    
    dict={}
    for tstamp in period:
        stamp_str="%04d%02d%02d_%02d"%(tstamp.year,tstamp.month,tstamp.day,tstamp.hour)
        dict=get_track_info(dict,sim,stamp_str)
        #except:
        #    pass
        
    #Dictionary built, saving to DataFrame:
    storms=pd.DataFrame.from_dict(dict,orient="index")
    print(storms.shape)
    hrs=(storms.shape[1] - 2)/8
    col_names=["area","bounds","clon","clat","PSu","PSv","tmin","tmean"]
    col_names=["storm_id","start_time"]+[col_name+"_%02d"%i for i in range(0,int(hrs)) for col_name in col_names]
    storms.columns=col_names
    storms=storms.set_index(storms["storm_id"])
    del storms["storm_id"]
    #Apply personal MCS criteria:
    if args.mcs:
        area_thld=5000
        areas=storms.filter(regex="area").astype("float")
        storms["mcs_thld"]=areas.where(areas>=area_thld).count(axis=1)
        storms["tmin_min"]=storms.filter(regex="tmin").min(axis=1)
        
        mcs_data=storms[(storms["tmin_min"]<223) & (storms["mcs_thld"]>=1)]
        mcs_data.to_csv(f"{output_dir}/{sim}_{region}_MCS_tracks.csv")
        
    elif arg.mcs=="MCSMIP": # DOESN'T INCLUDE RAIN AT THIS STAGE
        storms["tmin_min"]=storms.filter(regex="tmin").min(axis=1)
        
        area_thld=40000
        areas=storms.filter(regex="area").astype("float")
        storms["mcs_thld"]=areas.where(areas>=area_thld).count(axis=1)
        
        mcs_data=storms[(storms["tmin_min"]<227) & (storms["mcs_thld"]>=4)]
        mcs_data.to_csv(f"{output_dir}/{sim}_{region}_MCS_track_output.csv".format(sim))
        
    else:
        mcs_data.to_csv(f"{output_dir}/{sim}_{region}_all_tracks.csv")

    return mcs_data


def calc_mean_speeds(storm_data,sim):
    # Enables data to be passed direct:
    if len(storm_data)==0:
        storm_data=pd.read_csv(f"{output_dir}/{sim}_{region}_MCS_tracks.csv")
        
    prop_data = {"mcs":[],"ltime":[],"speeds":[],"dirs":[]}
    geo=cgeo.Geodesic()
    for i in range(len(storm_data)):
        storm_locs=storm_data.iloc[i].dropna()
        # Note use of threshold!
        is_mcs = storm_locs.filter(regex="area").where(storm_locs.filter(regex="area") > 5000).dropna()
        #Include cutoff to remove storms that only pop over MCS threshold once, can pollute tracking stats
        if len(is_mcs) > 1:
            prop_data["mcs"].append(storm_locs.name)
            prop_data["ltime"].append(len(is_mcs))

            #Get mean storm speeds; geodesic distance between start and end centroids / lifetime.
            storm_locs=storm_locs.filter(regex="cl")
            start_lon, start_lat = storm_locs.iloc[1], storm_locs.iloc[0]
            end_lon, end_lat = storm_locs.iloc[-1], storm_locs.iloc[-2]
            
            path=cgeo.Geodesic.inverse(geo,np.array((start_lon,start_lat)),np.array((end_lon,end_lat)))
        
            prop_data["speeds"].append(path[0,0] / (3600*len(storm_locs)/2))
            prop_data["dirs"].append(path[0,1])

    storm_prop=pd.DataFrame(prop_data)
    storm_prop.index = storm_prop["mcs"]
    storm_prop["dirs"][storm_prop["dirs"]<0] = storm_prop["dirs"][storm_prop["dirs"]<0] + 360

    storm_prop.to_csv(f"{output_dir}/{sim}_{region}_MCS_mean_props.csv",index=False)
    #return storm_prop



def pop_rains_masks(sim,mcs_tab=[]):
    if len(mcs_tab)==0:
        mcs_tab=pd.read_csv(f"{output_dir}/{sim}_{region}_MCS_tracks.csv")
        mcs_tab["start_time"]=pd.to_datetime(mcs_tab["start_time"].astype(str),format="%Y%m%d_%H")
    period=pd.date_range(mcs_tab.start_time.iloc[0],mcs_tab.start_time.iloc[-1],freq="H")
    # Storm ID's are changed vs raw tracks to enable consistent identification of masks
    mcs_tab["storm_id"]=np.arange(1,len(mcs_tab)+1)
    mcs_tab.index=mcs_tab["storm_id"]

    # Key lists to populate.
    # - keeps track of the mask dataArrays. Split on the timesteps within period. After initial creation a mask will be accessed and updated multiple times, from storms with later init times.
    mcs_masks=[]
    # - populates rain columns in the collated csv files. Split on start_time
    rain_vals=[]

    # Loop through unique start times
    for start_time in pd.to_datetime(mcs_tab.start_time.unique()):
        #Isolate storms which all initiated at same time:
        mcs_start_group=mcs_tab[mcs_tab["start_time"]==start_time].dropna(axis=1,how="all")
        print(start_time,": ",len(mcs_start_group)," storms") 
        
        # hrs eqn gets number of distinct hour timesteps in grouping. The OLR output csvs columns comprise:
        # - groups of 8 columns per timestep onward from different possible initialisation times
        # - 4 extra columns: storm_id, start_time, min storm T and mcs_thld (# times storm breached OLR size threshold)
        hrs=(mcs_start_group.shape[1] - 4)/8
        # Now loop through the lifetimes of the storms
        for hr in range(int(hrs)):
            #if hr % 3 == 0:
            #   print(hr)
            rain_vols,rain_maxes=np.zeros(len(mcs_start_group)),np.zeros(len(mcs_start_group))
            tmin_lats,tmin_lons=np.zeros(len(mcs_start_group)),np.zeros(len(mcs_start_group))
            loc=mcs_start_group.columns.to_list().index("tmean_%02d"%hr)

            #Isolate single timestep, hr hours onward from common start time:
            mcs_timestep=mcs_start_group.filter(regex="_%02d"%hr)
            timestamp=start_time+datetime.timedelta(hours=(hr))
            period_idx=int(np.where(period==timestamp)[0])

            # TAG TAG TAG
            ################################################################################
            #### !!! USER CUSTOMISATION REQUIRED !!! ###

            # Here we load in rainfall and TOA OLR data. So needs customising to the data source
            # End result needs to be TWO single-timestep, 2D FILES, precip + bt_mask
            # Source here is UM pp files, each 12 hours long
            
            date_str="%04d%02d%02d"%(timestamp.year,timestamp.month,timestamp.day)
            pfile=root+'precip/{}_{}_{}_precip_hourly.nc'.format(date_str,dymnd_run,domain)
            rfile=root+'single_olwr/{}_{}_{}_olwr_hourly.nc'.format(date_str,dymnd_run,domain)
            pfile=load_file(pfile,"precipitation_rate")
            bt=olr_to_bt(load_file(rfile,"toa_outgoing_longwave_flux"))
            bt=bt.where(bt<241).fillna(0)
            precip=pfile.isel(time=(timestamp.hour%24))
            ################################################################################

             # If timestep is new to algorithm, add new set of labelled data
            if period_idx >= len(mcs_masks):
                bt_mask=bt.isel(time=(timestamp.hour%24))
                # Label all distinct features meeting criteria. Structure is crucial to match tracker output accurately.
                labeled_array = label(bt_mask.values,structure=np.ones((3,3)))[0]
                #labeled_array = np.expand_dims(labeled_array,0)
                
            # If timestep has been encountered previously, load in the relevant masks, which are then edited.
            else:
                bt_mask=mcs_masks[period_idx]
                labeled_array=bt_mask.values

            for i in range(len(mcs_start_group)):
                # Get storm bounds - output direct from simpleTrack. Exception handles empty cell values, i.e. where storm has dissipated
                try:
                    bounds=[int(bound) for bound in mcs_timestep["bounds_%02d"%hr].iloc[i].split(",")]
                except:
                    rain_vols[i],rain_maxes[i]=np.nan,np.nan
                    tmin_lats[i],tmin_lons[i]=np.nan,np.nan
                    continue
                    
                id=mcs_timestep.index[i]
                # Get the region of mask which corresponds to storm "id". (0,1) gives top LEFT coordinate of domain; 2 gives width; 3 gives height (l572 of object_tracking.py).
                # Thus require lat/lon form data
                storm_zoom = labeled_array[(bounds[1]-bounds[3]):bounds[1]+1,bounds[0]:(bounds[0]+bounds[2])+1]
                # Get most common non-zero label in storm_zoom:
                lab=stats.mode(np.where(storm_zoom!=0,storm_zoom,np.nan),axis=None,nan_policy="omit",keepdims=True)[0]
                
                # There are some instances where a bound width/height = 0, so no result in lab. Catch prevents a fatal error.
                if len(lab)>0:
                #if storm_zoom.size > len(storm_zoom):
                    # Catch for just in case lab does not identify a storm label; mode on all nans gives result 0
                    if lab==0:
                        rain_vols[i],rain_maxes[i]=np.nan,np.nan
                        tmin_lats[i],tmin_lons[i]=np.nan,np.nan
                        print("No storms")
                    else:
                        # Replace identified label with storm id - LARGE PAD; crucial to avoid mix up with the labels set in l143. Being -ve enables easy extraction of storm_ids later on
                        # In this manner labeled_array builds up pictures of our storms if needed
                        labeled_array=np.where(labeled_array==lab,-9999999+id,labeled_array)
                        # Get the rainfall values within the identified storm area:
                        if len(labeled_array[labeled_array==-9999999+id].flatten()) == 0:
                            print("No labels found, error",hr,lab,id)
                            break
                            
                        storm_precip=precip.where(np.where(labeled_array==-9999999+id,1,0)>0)
                        # Extract total and max storm rainfall at this timestep
                        rain_vols[i]=float(storm_precip.sum())
                        rain_maxes[i]=float(storm_precip.max())
                        #print(float(storm_precip.sum()),float(storm_precip.max()))

                        storm_bt=bt_mask.where(np.where(labeled_array==-9999999+id,1,0)>0)
                        argmins=np.where(storm_bt.values==float(storm_bt.min()))
                        try:
                            tmin_lats[i],tmin_lons[i]=float(storm_bt[ycoord][argmins[0]]),float(storm_bt[xcoord][argmins[1]])
                        except:
                            tmin_lats[i],tmin_lons[i]=np.nan,np.nan
                else:
                    print("Missing dimension")
                    rain_vols[i],rain_maxes[i]=np.nan,np.nan
                    tmin_lats[i],tmin_lons[i]=np.nan,np.nan
                

            # Add rain values into the csv component:
            mcs_start_group.insert(loc+1,"rain_vol_%02d"%hr,rain_vols)
            mcs_start_group.insert(loc+2,"rain_max_%02d"%hr,rain_maxes)
            mcs_start_group.insert(loc,"tmin_lat_%02d"%hr,tmin_lats)
            mcs_start_group.insert(loc,"tmin_lon_%02d"%hr,tmin_lons)

            if mask_out:
                # Get storm masks into a dataArray, then output appropriately based on whether new or previous time.
                bt_mask.values=labeled_array
                if period_idx >= len(mcs_masks):
                    mcs_masks.append(bt_mask.rename("mcs_mask"))
                else:
                    mcs_masks[period_idx]=bt_mask
        #print(mcs_start_group)
        #print(mcs_start_group.filter(regex="rain"))
        rain_vals.append(mcs_start_group)

    
    if mask_out:
        mcs_masks=xr.concat(mcs_masks,dim="time")
        # Removes all remnant non-MCS label items and then corrects for the large padding:
        mcs_masks=mcs_masks.where(mcs_masks<0) + 9999999
        mcs_masks=mcs_masks.assign_attrs(units="unitless",long_name="MCS mask with track number")
        mcs_masks.to_netcdf(f"{output_dir}/{sim}_{region}_MCS_track_masks.nc")
    
    rain_vals=pd.concat(rain_vals,axis=0)
    rain_vals[["temp1","temp2"]]=rain_vals[["tmin_min","mcs_thld"]]
    rain_vals=rain_vals.drop(columns=["tmin_min","mcs_thld"]).rename(columns={"temp1":"tmin_min","temp2":"mcs_thld"})
    rain_vals.to_csv(f"{output_dir}/{sim}_{region}_MCS_tracks_rain.csv",index=False)
    
    return rain_vals

        
###########################################################################################################################################################

# TAG TAG TAG
################################################################################
#### !!! USER CUSTOMISATION REQUIRED !!! ###
# Edit first argparse arguments to reflect individual data. KEEP LAST 3 ARGUMENTS
# -update = blank to make initial table; "rains" to add rain data and make MCS masks; "mean_prop" for lifetime mean speeds
# -mcs = Boolean for applying MCS criteria; default true
# -mask = Boolean for making MCS mask netcdf files; slows things down a bit so default False 

#Available regions: "sahel", "wafrica", "ea-waf", "safrica", "samerica", "india", "aus"
#Available simulations: "channel", "lam", "global"
#Available resolutions: "n1280", "n2560", "km4p4", "km2p2"
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--region", required=True, type=str)
parser.add_argument("-d", "--domain", required=False, default="lam", type=str)
parser.add_argument("-res", "--res", required=False, default="km2p2", type=str)
parser.add_argument("-c", "--config", required=False, default="RAL3p2", type=str)
parser.add_argument("-dm", "--driving_model", required=False, type=str, default="GAL9")
parser.add_argument("-u", "--update", required=False, type=str)
parser.add_argument("-mcs", "--mcs", required=False, type=bool, default=True)
parser.add_argument("-mask", "--mask", required=False, type=bool, default=False)
args = parser.parse_args()

region=args.region.lower()
domain=args.domain.lower()
res_name=args.res
config=args.config
dm=args.driving_model
mask_out=args.mask

raw_track_dir=raw_track_dir+f"{domain}_{res_name}_{config}_{region}/"

#For N. Hemisphere regions want summer DYMAMOND period
if region == "sahel" or region == "wafrica" or region == "india" or region=="ea-waf" or region=="summer":
    period=pd.date_range("2016-08-01 00:00","2016-09-09 23:00",freq="H")
    dymnd_run="20160801T0000Z"
if region == "samerica" or region == "safrica" or region == "aus" or region=="winter":
    period=pd.date_range("2020-01-20 00:00","2016-02-28 23:00",freq="H")
    dymnd_run="20200120T0000Z"

# NEED TO LOAD any representative file that has same grid as what was used for tracking
# simpleTrack outputs give coordinate information as indices on numpy grid. Use reference file to convert to lat/lon:
if region == "sahel" or region =="wafrica":
    ds_ref=xr.open_dataset("/gws/nopw/j04/kscale/DATA/outdir_20160801T0000Z/DMn1280GAL9/lam_africa_km2p2_RAL3p2/single_olwr/20160830_20160801T0000Z_africa_olwr_hourly.nc"
                      ).sel(latitude=slice(-6,24),longitude=slice(-18,32))
    ref_lats=ds_ref.latitude.values
    ref_lons=ds_ref.longitude.values
elif region=="ea-waf":
    ds_ref=xr.open_dataset("/gws/nopw/j04/kscale/DATA/outdir_20160801T0000Z/DMn1280GAL9/channel_n2560_RAL3p2/single_olwr/20160830_20160801T0000Z_channel_olwr_hourly.nc"
                      ).sel(latitude=slice(0,25),longitude=slice(-40,30))
    ref_lats=ds_ref.latitude.values
    ref_lons=ds_ref.longitude.values
elif region=="winter" or region=="summer":
    ds_ref=xr.open_dataset("/gws/nopw/j04/kscale/DATA/outdir_20160801T0000Z/DMn1280GAL9/channel_n2560_RAL3p2/single_olwr/20160830_20160801T0000Z_channel_olwr_hourly.nc"
                      ).sel(latitude=slice(-35,25))
    ref_lats=ds_ref.latitude.values
    ref_lons=ds_ref.longitude.values


if domain!="lam":
    root='/gws/nopw/j04/kscale/DATA/outdir_{}/DMn1280{}/{}_{}_{}/'.format(dymnd_run,dm,domain,res_name,config)
else:
    if region == "wafrica" or region == "safrica" or region == "sahel":
        root='/gws/nopw/j04/kscale/DATA/outdir_{}/DMn1280{}/lam_africa_{}_{}/'.format(dymnd_run,dm,res_name,config)
        domain="africa"
    elif region == "aus":
        root='/gws/nopw/j04/kscale/DATA/outdir_{}/DMn1280{}/lam_sea_{}_{}/'.format(dymnd_run,dm,res_name,config)
        domain="sea"
    else:
        root='/gws/nopw/j04/kscale/DATA/outdir_{}/DMn1280{}/lam_{}_{}_{}/'.format(dymnd_run,dm,region,res_name,config)
        domain=region

sim=args.domain+"_"+res_name+"_"+config

if args.update is None:
    build_table(sim)
elif args.update == "rains":
    pop_rains_masks(sim)
elif args.update == "mean_prop":
    calc_mean_speeds([],sim)
