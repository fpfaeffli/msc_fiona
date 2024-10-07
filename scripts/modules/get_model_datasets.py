#%%
import os
import glob
import sys
import numpy as np
import xarray as xr
import pandas as pd
import scipy.spatial.qhull as qhull
import re
import glob

#%%

class ModelGetter():
    """
    author: Eike E. Koehn
    date: June 7, 2024
    description: This class contains a number of functions that enable the loading of simulation data. 
    """ 

    @staticmethod
    def define_model_setups():
        """
        author: Eike E. Koehn
        date: June 10, 2024
        description: Specify the different model setups.
        """
        configs = ['roms_only','romsoc_fully_coupled']
        scenarios = ['present','ssp245','ssp585']
        simulation_types = ['spinup','hindcast']
        parent_models = ['mpi-esm1-2-hr']
        ensemble_runs = ['000','001']    # ensemble runs within a certain config/scenario/parent_model combination
        return configs, scenarios, simulation_types, parent_models, ensemble_runs
    
    @staticmethod
    def define_output_formats():
        """
        author: Eike E. Koehn
        date: June 10, 2024
        description: Specify the different output formats.
        """        
        temp_resolutions = ['daily','monthly']
        vert_structs = ['avg','zavg']
        vtypes = ['oceanic','atmospheric']
        return temp_resolutions, vert_structs, vtypes


    #######################################################################
    ## NOW THE METHODS THAT ALLOW THE RETRIEVAL OF THE MODEL SIMULATIONS ##
    #######################################################################

    @staticmethod
    def get_model_output_paths(config,scenario,simulation_type,ensemble_run,temp_resolution,vert_struct,parent_model=None,vtype=None):
        """
        author: Eike E. Koehn
        date: June 10, 2024
        description: Specify the different model paths.
        if config == 'romsoc_fully_coupled', need to specify if vtype is 'oceanic' or 'atmospheric'
        if scenario != 'present', need to specify parent_model (e.g. 'mpi-esm1-2-hr')
        """        
        root_dir = "/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/"
        # Specify the paths for the ROMS-only setup
        if config == 'roms_only':
            if scenario == 'present':
                if simulation_type == 'spinup':
                    if ensemble_run == '000':
                        if temp_resolution == 'monthly':
                            if vert_struct == 'avg':
                                model_path = f"{root_dir}spinup_roms/present/monthly/avg/"
                            elif vert_struct == 'zavg':
                                model_path = f"{root_dir}spinup_roms/present/monthly/z_avg/"
                        elif temp_resolution == 'daily':
                            model_path = 'nan'
                    elif ensemble_run != '000':
                        model_path = 'nan'
                elif simulation_type == 'hindcast':
                    if ensemble_run == '000':
                        if temp_resolution == 'monthly':
                            if vert_struct == 'avg':
                                model_path =  f"{root_dir}roms_only/present/monthly/avg/"
                            elif vert_struct == 'zavg':
                                model_path =  f"{root_dir}roms_only/present/monthly/z_avg/"
                        elif temp_resolution == 'daily':
                            if vert_struct == 'avg':
                                model_path =  f"{root_dir}roms_only/present/daily/avg/"
                            elif vert_struct == 'zavg':
                                model_path =  f"{root_dir}roms_only/present/daily/z_avg/"                          
                    elif ensemble_run != '000':
                        model_path = 'nan'
            elif scenario == 'ssp245':
                if parent_model == None:
                    raise Exception('No parent model specified. Need to pass a string for parent model variable, as a future projection is requested.')
                elif parent_model == 'mpi-esm1-2-hr':
                    if simulation_type == 'spinup':
                        if ensemble_run == '000':
                            if temp_resolution == 'monthly':
                                if vert_struct == 'avg':
                                    model_path = f"{root_dir}spinup_roms/ssp245/monthly/avg/"
                                elif vert_struct == 'zavg':
                                    model_path = f"{root_dir}spinup_roms/ssp245/monthly/z_avg/"
                            elif temp_resolution == 'daily':
                                model_path = 'nan'
                        elif ensemble_run != '000':
                            model_path = 'nan'
                    elif simulation_type == 'hindcast':
                        if ensemble_run == '000':
                            if temp_resolution == 'monthly':
                                if vert_struct == 'avg':
                                    model_path =  f"{root_dir}roms_only/ssp245/monthly/avg/"
                                elif vert_struct == 'zavg':
                                    model_path =  f"{root_dir}roms_only/ssp245/monthly/z_avg/"
                            elif temp_resolution == 'daily':
                                if vert_struct == 'avg':
                                    model_path =  f"{root_dir}roms_only/ssp245/daily/avg/"
                                elif vert_struct == 'zavg':
                                    model_path = f"{root_dir}roms_only/ssp245/daily/z_avg/"                        
                        elif ensemble_run != '000':
                            model_path = 'nan'
            elif scenario == 'ssp585':
                if parent_model == None:
                    raise Exception('No parent model specified. Need to pass a string for parent model variable, as a future projection is requested.')
                elif parent_model == 'mpi-esm1-2-hr':
                    if simulation_type == 'spinup':
                        if ensemble_run == '000':
                            if temp_resolution == 'monthly':
                                if vert_struct == 'avg':
                                    model_path = f"{root_dir}spinup_roms/ssp585/monthly/avg/"
                                elif vert_struct == 'zavg':
                                    model_path = f"{root_dir}spinup_roms/ssp585/monthly/z_avg/"
                            elif temp_resolution == 'daily':
                                model_path = 'nan'
                        elif ensemble_run != '000':
                            model_path = 'nan'
                    elif simulation_type == 'hindcast':
                        if ensemble_run == '000':
                            if temp_resolution == 'monthly':
                                if vert_struct == 'avg':
                                    model_path =  f"{root_dir}roms_only/ssp585/monthly/avg/"
                                elif vert_struct == 'zavg':
                                    model_path =  f"{root_dir}roms_only/ssp585/monthly/z_avg/"
                            elif temp_resolution == 'daily':
                                if vert_struct == 'avg':
                                    model_path =  f"{root_dir}roms_only/ssp585/daily/avg/"
                                elif vert_struct == 'zavg':
                                    model_path =  f"{root_dir}roms_only/ssp585/daily/z_avg/"                        
                        elif ensemble_run != '000':
                            model_path = 'nan'         
        # Specify the paths for the ROMSOC setup, where i need to additionally specify whether i refer to atmospheric or ocean model
        elif config == 'romsoc_fully_coupled':
            if scenario == 'present':
                if simulation_type == 'spinup':
                    model_path = 'nan'
                elif simulation_type == 'hindcast':
                    if ensemble_run == '000':
                        if temp_resolution == 'monthly':
                            if vert_struct == 'avg':
                                if vtype == 'oceanic':
                                    model_path =  f"{root_dir}romsoc_fully_coupled/present/monthly/avg/"
                                elif vtype == 'atmospheric':
                                    model_path =  'nan'                     
                            elif vert_struct == 'zavg':
                                if vtype == 'oceanic':
                                    model_path =  f"{root_dir}romsoc_fully_coupled/present/monthly/z_avg/"
                                elif vtype == 'atmospheric':
                                    model_path = 'nan'
                        elif temp_resolution == 'daily':
                            if vert_struct == 'avg':
                                if vtype == 'oceanic':
                                    model_path = f"{root_dir}romsoc_fully_coupled/present/daily/avg/" 
                                elif vtype == 'atmospheric':
                                    model_path =  f"{root_dir}romsoc_fully_coupled/atmosphere/present/daily/"
                            elif vert_struct == 'zavg':
                                if vtype == 'oceanic':
                                    model_path =  f"{root_dir}romsoc_fully_coupled/present/daily/z_avg/"
                                elif vtype == 'atmospheric':
                                    model_path = 'nan' 
                    elif ensemble_run == '001':
                        if temp_resolution == 'monthly':
                            if vert_struct == 'avg':
                                if vtype == 'oceanic':
                                    model_path =  f"{root_dir}romsoc_fully_coupled/ensemble_members/present/ens001/monthly/avg/"
                                elif vtype == 'atmospheric':
                                    model_path =  'nan'                     
                            elif vert_struct == 'zavg':
                                if vtype == 'oceanic':
                                    model_path =  f"{root_dir}romsoc_fully_coupled/ensemble_members/present/ens001/monthly/z_avg/"
                                elif vtype == 'atmospheric':
                                    model_path = 'nan'
                        elif temp_resolution == 'daily':
                            if vert_struct == 'avg':
                                if vtype == 'oceanic':
                                    model_path = f"{root_dir}romsoc_fully_coupled/ensemble_members/present/ens001/daily/avg/" 
                                elif vtype == 'atmospheric':
                                    model_path =  f"{root_dir}romsoc_fully_coupled/atmosphere/ensemble_members/present/ens001/daily/"
                            elif vert_struct == 'zavg':
                                if vtype == 'oceanic':
                                    model_path =  f"{root_dir}romsoc_fully_coupled/ensemble_members/present/ens001/daily/z_avg/"
                                elif vtype == 'atmospheric':
                                    model_path = 'nan'                         
                    else: 
                        model_path = 'nan'
            elif scenario == 'ssp245':
                if parent_model == None:
                    raise Exception('No parent model specified. Need to pass a string for parent model variable, as a future projection is requested.')
                elif parent_model == 'mpi-esm1-2-hr':                    
                    if simulation_type == 'spinup':
                        model_path = 'nan'
                    elif simulation_type == 'hindcast':
                        if ensemble_run == '000':
                            if temp_resolution == 'monthly':
                                if vert_struct == 'avg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ssp245/monthly/avg/"
                                    elif vtype == 'atmospheric':
                                        model_path = 'nan'
                                elif vert_struct == 'zavg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ssp245/monthly/z_avg/"
                                    elif vtype == 'atmospheric':
                                        model_path = 'nan'
                            elif temp_resolution == 'daily':
                                if vert_struct == 'avg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ssp245/daily/avg/"
                                    elif vtype == 'atmospheric':
                                        model_path = f"{root_dir}romsoc_fully_coupled/atmosphere/ssp245/daily/"
                                elif vert_struct == 'zavg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ssp245/daily/z_avg/"
                                    elif vtype == 'atmospheric':
                                        model_path = 'nan'                      
                        elif ensemble_run == '001':
                            if temp_resolution == 'monthly':
                                if vert_struct == 'avg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ensemble_members/ssp245/ens001/monthly/avg/"
                                    elif vtype == 'atmospheric':
                                        model_path =  'nan'                     
                                elif vert_struct == 'zavg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ensemble_members/ssp245/ens001/monthly/z_avg/"
                                    elif vtype == 'atmospheric':
                                        model_path = 'nan'
                            elif temp_resolution == 'daily':
                                if vert_struct == 'avg':
                                    if vtype == 'oceanic':
                                        model_path = f"{root_dir}romsoc_fully_coupled/ensemble_members/ssp245/ens001/daily/avg/" 
                                    elif vtype == 'atmospheric':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/atmosphere/ensemble_members/ssp245/ens001/daily/"
                                elif vert_struct == 'zavg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ensemble_members/ssp245/ens001/daily/z_avg/"
                                    elif vtype == 'atmospheric':
                                        model_path = 'nan'                         
                        else: 
                            model_path = 'nan'
            elif scenario == 'ssp585':
                if parent_model == None:
                    raise Exception('No parent model specified. Need to pass a string for parent model variable, as a future projection is requested.')
                elif parent_model == 'mpi-esm1-2-hr':                    
                    if simulation_type == 'spinup':
                        model_path = 'nan'
                    elif simulation_type == 'hindcast':
                        if ensemble_run == '000':
                            if temp_resolution == 'monthly':
                                if vert_struct == 'avg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ssp585/monthly/avg/"
                                    elif vtype == 'atmospheric':
                                        model_path = 'nan'
                                elif vert_struct == 'zavg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ssp585/monthly/z_avg/"
                                    elif vtype == 'atmospheric':
                                        model_path = 'nan'
                            elif temp_resolution == 'daily':
                                if vert_struct == 'avg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ssp585/daily/avg/"
                                    elif vtype == 'atmospheric':
                                        model_path = f"{root_dir}romsoc_fully_coupled/atmosphere/ssp585/daily/"
                                elif vert_struct == 'zavg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ssp585/daily/z_avg/"
                                    elif vtype == 'atmospheric':
                                        model_path = 'nan'                      
                        elif ensemble_run == '001':
                            if temp_resolution == 'monthly':
                                if vert_struct == 'avg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ensemble_members/ssp585/ens001/monthly/avg/"
                                    elif vtype == 'atmospheric':
                                        model_path =  'nan'                     
                                elif vert_struct == 'zavg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ensemble_members/ssp585/ens001/monthly/z_avg/"
                                    elif vtype == 'atmospheric':
                                        model_path = 'nan'
                            elif temp_resolution == 'daily':
                                if vert_struct == 'avg':
                                    if vtype == 'oceanic':
                                        model_path = f"{root_dir}romsoc_fully_coupled/ensemble_members/ssp585/ens001/daily/avg/" 
                                    elif vtype == 'atmospheric':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/atmosphere/ensemble_members/ssp585/ens001/daily/"
                                elif vert_struct == 'zavg':
                                    if vtype == 'oceanic':
                                        model_path =  f"{root_dir}romsoc_fully_coupled/ensemble_members/ssp585/ens001/daily/z_avg/"
                                    elif vtype == 'atmospheric':
                                        model_path = 'nan'                         
                        else: 
                            model_path = 'nan'
        return model_path

    @staticmethod
    def get_model_filenames(model_path,simulation_type,hindcast_start_year=2011,vtype='oceanic',vtype_extra=None):
        """
        author: Eike E. Koehn
        date: June 10, 2024
        description: This function gets the sorted filenames that are needed for the model_path and removes files that are not needed.
        """
        print(f'Getting files from {model_path}.')
        if model_path != 'nan':
            # Get the list of filenames for the respective path
            if vtype == 'oceanic':
                filenames_list = sorted(glob.glob(f"{model_path}/*.nc"))
            elif vtype == 'atmospheric':
                if vtype_extra == None:
                    filenames_list = sorted(glob.glob(f"{model_path}/*[0-9].nc"))
                elif vtype_extra == 'clouds' or 'pressure':
                    filenames_list = sorted(glob.glob(f"{model_path}/*{vtype_extra}.nc"))
            # Do some case dependent postprocessing (e.g., taking out certain years such as 2010)
            filtered_filenames = []
            for filename in filenames_list:
                if vtype == 'oceanic':
                    match = re.search(r'(20\d{2})', filename.split('/')[-1])
                elif vtype == 'atmospheric':
                    match = re.search(r'(lffd20\d{2})',filename.split('/')[-1])
                if match:
                    if vtype == 'oceanic':
                        year = int(match.group(1))
                    elif vtype == 'atmospheric':
                        year = int(match.group(1)[-4:])
                    if simulation_type == 'hindcast':
                        if year >= hindcast_start_year:
                            print(filename)
                            filtered_filenames.append(filename)
                    elif simulation_type == 'spinup':
                        if year < hindcast_start_year:
                            print(filename)
                            filtered_filenames.append(filename)
        else:
            filtered_filenames = []
        return filtered_filenames

    @staticmethod
    def get_model_dataset(config,scenario,simulation_type,ensemble_run,temp_resolution,vert_struct,parent_model=None,vtype='oceanic',vtype_extra=None):
        """
        author: Eike E. Koehn
        date: June 10, 2024
        description: Loading the dataset for a particular model run.
        """        
        print('Get the model dataset.')
        model_path = ModelGetter.get_model_output_paths(config,scenario,simulation_type,ensemble_run,temp_resolution,vert_struct,parent_model=parent_model,vtype=vtype)
        #print(model_path)
        #print(simulation_type)
        model_filenames = ModelGetter.get_model_filenames(model_path,simulation_type,vtype=vtype,vtype_extra=vtype_extra)  
        #print('model_filenames:')
        #print(model_filenames)
        #print('---')
        model_dataset = xr.open_mfdataset(model_filenames,concat_dim='time',combine='nested',parallel=True)
        # fix the time dimensions/coordinates
        if simulation_type == 'hindcast':
            if temp_resolution == 'monthly':
                time = pd.date_range('2011-01-01', periods=model_dataset.dims['time'],freq='1MS')
            elif temp_resolution == 'daily':
                time = pd.date_range('2011-01-01', periods=model_dataset.dims['time'],freq='1D')
        elif simulation_type == 'spinup':
            if temp_resolution == 'monthly':
                time = pd.date_range('2000-01-01', periods=model_dataset.dims['time'],freq='1MS')
            elif temp_resolution == 'daily':
                raise Exception('Daily spinup data should not exist.')
        # Check if 'time' coordinate exists, and add if it doesn't. Otherwise overwrite with the time range
        if 'time' not in model_dataset.coords:
            # Create a time coordinate with a specified range
            # For this example, we'll create a simple range of dates
            model_dataset = model_dataset.assign_coords(time=('time', time))
        else:
            model_dataset['time'] = time

        # fix the horizontal dimensions
        if vtype == 'oceanic':
            if 'lon_rho' not in model_dataset.coords:
                model_lon,model_lat = ModelGetter.get_model_coords()
                model_dataset = model_dataset.assign_coords(lon=(['eta_rho','xi_rho'], model_lon.data))
            else:
                model_dataset = model_dataset.rename({'lon_rho':'lon'})
            if 'lat_rho' not in model_dataset.coords:
                model_lon,model_lat = ModelGetter.get_model_coords()
                model_dataset = model_dataset.assign_coords(lat=(['eta_rho','xi_rho'], model_lat.data))    
            else:
                model_dataset = model_dataset.rename({'lat_rho':'lat'})                    
        return model_dataset

    ###########################################################################
    ## NOW THE METHODS THAT CONCERN THE GRID, MASK, COORDS, AREA, DIST2COAST ##
    ###########################################################################

    @staticmethod
    def get_model_mask(vtype='oceanic',mask_grid='rho'):
        """
        author: Eike E. Koehn
        date: June 10, 2024
        description: Getting the ocean mask of a model.
        """
        if vtype == 'oceanic':
            ds_grd = xr.open_dataset("/nfs/kryo/work/loher/ROMS/pactcs30/grd/pactcs30_grd.nc")
            if mask_grid == 'rho':
                mask = ds_grd.mask_rho
            elif mask_grid == 'u':
                mask = ds_grd.mask_u
            elif mask_grid == 'v':
                mask = ds_grd.mask_v
        elif vtype == 'atmospheric':
            ds_grd = xr.open_dataset('/nfs/meso/work/loher/ROMSOC/ROMSOC_Pacific2021_era5frc_COSMOorg_GPU_newcoupledhindcast/masks.nc')
            if mask_grid == 'rho':
                mask = ds_grd['cosp.msk']
            elif mask_grid == 'u':
                mask = ds_grd['cosu.msk']
            elif mask_grid == 'v':
                mask = ds_grd['cosv.msk']
        mask = xr.where(mask==0,np.NaN,mask)
        return mask

    @staticmethod
    def get_model_coords(vtype='oceanic'):
        """
        author: Eike E. Koehn
        date: June 10, 2024
        description: Getting the coordinates of a model (on the rho grid).
        """
        if vtype == 'oceanic':
            ds_grd = xr.open_dataset("/nfs/kryo/work/loher/ROMS/pactcs30/grd/pactcs30_grd.nc")
            lon = ds_grd.lon_rho
            lat = ds_grd.lat_rho
        elif vtype == 'atmospheric':
            raise Exception('atmospheric model mask not yet implemented')
        return lon,lat
    
    @staticmethod
    def get_model_area(vtype='oceanic',area_grid='rho'):
        """
        author: Eike E. Koehn
        date: June 10, 2024
        description: Getting the area of a model grid cell in m^2.
        """
        if vtype == 'oceanic':
            ds_grd = xr.open_dataset("/nfs/kryo/work/loher/ROMS/pactcs30/grd/pactcs30_grd.nc")
            if area_grid == 'rho':
                area = (1/ds_grd.pm)*(1/ds_grd.pn) # in m2
                area = xr.where(ds_grd.mask_rho==0,np.NaN,area)
                area = area.rename({'lon_rho':'lon','lat_rho':'lat'})
        elif vtype == 'atmospheric':
            ds_grd = xr.open_dataset("/nfs/meso/work/loher/ROMSOC/ROMSOC_Pacific2021_era5frc_COSMOorg_GPU_newcoupledhindcast/areas.nc")
            if area_grid == 'rho':
                area = ds_grd['cosp.srf']
            elif area_grid == 'u':
                area = ds_grd['cosu.srf']
            elif area_grid == 'v':
                area = ds_grd['cosv.srf']
        return area

    @staticmethod
    def get_distance_to_coast(vtype='oceanic'):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: Getting the distance to the coast (at the surface) for a model grid cell.
        """
        if vtype == 'oceanic':
            ds_dist = xr.open_dataset('/nfs/kryo/work/martinfr/Data/pactcs30/pactcs30_dist2coast.nc')
            dcoast = ds_dist.dcoast # in km
            dcoast = dcoast.rename({'lon_rho':'lon','lat_rho':'lat'})
        elif vtype == 'atmospheric':
            raise Exception('atmospheric model d2coast not yet implemented')
        return dcoast
    
    ####################################################################
    ## NOW THE METHODS THAT CONCERN THE CLIMATOLOGY & EXTREME THRESHOLDS
    ####################################################################

    @staticmethod
    def get_threshold_climatology_dataset(config,scenario,varia,threshold_type,threshold_value,nzlevs=37,analysis_start_year=2011,analysis_end_year=2021,baseperiod_start_year=2011,baseperiod_end_year=2021,aggregation_kernel=11,smoothing_kernel=31):
        if threshold_type == 'relative':
            root_dir = "/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/extreme_analysis/thresholds_and_climatology/"
            path_name = '{}{}/{}/'.format(root_dir,config,scenario) 
            file_name = f'hobday2016_threshold_and_climatology_{varia}_{nzlevs}zlevs_full_1x1meanpool_downsampling_{analysis_start_year}-{analysis_end_year}analysisperiod_{threshold_value}perc_{baseperiod_start_year}-{baseperiod_end_year}baseperiod_fixedbaseline_{aggregation_kernel}aggregation_{smoothing_kernel}smoothing.nc'
            fn = xr.open_dataset(path_name+file_name)
        elif threshold_type == 'absolute':
            raise Exception('Not yet implemented.')
        return fn

    @staticmethod
    def include_feb29(data_365):
        if np.size(data_365.time)==365:
            data_365['time'] = pd.date_range('2001-01-01','2001-12-31')
            # Calculate the mean of 28th February and 1st March
            feb29 = (data_365.sel(time="2001-02-28") + data_365.sel(time="2001-03-01")) / 2
            feb29['time'] = pd.to_datetime("2004-02-29")
            # part1 and part 2
            part1 = data_365.sel(time=slice(None,"2001-02-28"))
            part1["time"] = pd.date_range("2004-01-01","2004-02-28")
            part2 = data_365.sel(time=slice("2001-03-01",None))
            part2["time"] = pd.date_range("2004-03-01","2004-12-31")
            # concatenate the data
            data_366 = xr.concat([part1,feb29,part2], dim="time")
            data_366['time'] = np.arange(366)
        # data_feb29 = np.mean(data_365[59:61,...],axis=0,keepdims=True)
        # data_366 = np.concatenate((data_365[:60,...],data_feb29,data_365[60:,...]),axis=0)
        return data_366

    @staticmethod
    def get_threshold(variable,depth_level,threshold_type,threshold_value,config,scenario):#,simulation_type,ensemble_run,temp_resolution,vert_struct,parent_model=None,vtype=None):
        fn = ModelGetter.get_threshold_climatology_dataset(config,scenario,variable,threshold_type,threshold_value)
        threshold = fn.thresh_smoothed.sel(depth=depth_level)#[:,depth_idx,...].values
        threshold_366 = ModelGetter.include_feb29(threshold)
        return threshold, threshold_366
    
    @staticmethod
    def get_climatology(variable,depth_level,threshold_type,threshold_value,config,scenario):#,simulation_type,ensemble_run,temp_resolution,vert_struct,parent_model=None,vtype=None):
        fn = ModelGetter.get_threshold_climatology_dataset(config,scenario,variable,threshold_type,threshold_value)
        climatology = fn.clim_smoothed.sel(depth=depth_level)
        climatology_366 = ModelGetter.include_feb29(climatology)
        return climatology, climatology_366
    
    @staticmethod
    def get_intensity_normalizer(variable,depth_level,threshold_type,threshold_value,config,scenario):#,simulation_type,ensemble_run,temp_resolution,vert_struct,parent_model=None,vtype=None):
        fn = ModelGetter.get_threshold_climatology_dataset(config,scenario,variable,threshold_type,threshold_value)
        intensity_normalizer = fn.intensity_normalizer_smoothed.sel(depth=depth_level)
        intensity_normalizer_366 = ModelGetter.include_feb29(intensity_normalizer)
        return intensity_normalizer, intensity_normalizer_366

    @staticmethod
    def concatenate_yearly_arrays(yearly_array,yearly_array_leap_year,start_year=2011,end_year=2021):
        year_range = np.arange(start_year,end_year+1)
        yearly_files_list = []
        for year in year_range:
            if np.mod(year,4)==0:
                da = yearly_array_leap_year
            else:
                da = yearly_array
            da['time'] = pd.date_range(f'{year}-01-01',f'{year}-12-31')                
            yearly_files_list.append(da)
        concatenated_data = xr.concat(yearly_files_list,dim='time')
        concatenated_data['time'] = pd.date_range(f'{start_year}-01-01',f'{end_year}-12-31')
        concatenated_data = concatenated_data.rename({'lat':'eta_rho','lon':'xi_rho'})
        return concatenated_data


########################################################################################################
#%% THIS IS A CLASS FOR LOADING IN THE CARBONATE CHEMISTRY DATA SETS, i.e. THE SATURATION HORIZONS
########################################################################################################

class ModelCarbonateChemistryGetter():
    """
    author: Eike E. Koehn
    date: June 7, 2024
    description: This class contains a number of functions that enable the loading of carbonate chemistry variables. 
    """ 
    
    @staticmethod
    def get_isosurface_paths_and_files(config,scenario,simulation_type,ensemble_run,temp_resolution,parent_model=None):
        """
        author: Eike E. Koehn
        date: June 10, 2024
        description: Specify the different model paths.
        if config == 'romsoc_fully_coupled', need to specify if vtype is 'oceanic' or 'atmospheric'
        if scenario != 'present', need to specify parent_model (e.g. 'mpi-esm1-2-hr')
        """        
        root_dir = "/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/processed_model_data/isosurfaces/"
        file_name = f'isosurface_omega_arag_offl_{temp_resolution}.nc'
        # Specify the paths for the ROMS-only setup
        if config == 'roms_only':
            if scenario == 'present':
                if simulation_type == 'spinup':
                    raise Exception('No isosurface file available.')
                elif simulation_type == 'hindcast':
                    if ensemble_run == '000':
                        if temp_resolution == 'monthly':
                            raise Exception('No isosurface file available.')
                        elif temp_resolution == 'daily':
                            model_path =  f"{root_dir}{config}/{scenario}/"                          
                    elif ensemble_run != '000':
                        raise Exception('No isosurface file available.')
            elif scenario == 'ssp245' or scenario == 'ssp585':
                if parent_model == None:
                    raise Exception('No parent model specified. Need to pass a string for parent model variable, as a future projection is requested.')
                elif parent_model == 'mpi-esm1-2-hr':
                    if simulation_type == 'spinup':
                        raise Exception('No isosurface file available.')
                    elif simulation_type == 'hindcast':
                        if ensemble_run == '000':
                            if temp_resolution == 'monthly':
                                raise Exception('No isosurface file available.')
                            elif temp_resolution == 'daily':
                                model_path =  f"{root_dir}{config}/{scenario}/"                          
                        elif ensemble_run != '000':
                            raise Exception('No isosurface file available.')       
        # Specify the paths for the ROMSOC setup, where i need to additionally specify whether i refer to atmospheric or ocean model
        elif config == 'romsoc_fully_coupled':
            if scenario == 'present':
                if simulation_type == 'spinup':
                    raise Exception('No isosurface file available.')       
                elif simulation_type == 'hindcast':
                    if ensemble_run == '000':
                        if temp_resolution == 'monthly':
                            raise Exception('No isosurface file available.')
                        elif temp_resolution == 'daily':
                            model_path =  f"{root_dir}{config}/{scenario}/"                          
                    elif ensemble_run == '001':
                        if temp_resolution == 'monthly':
                            raise Exception('No isosurface file available.')
                        elif temp_resolution == 'daily':
                            model_path =  f"{root_dir}{config}/ensemble_members/{scenario}/ens{ensemble_run}/"                                             
                    else: 
                        raise Exception('No isosurface file available.')       
            elif scenario == 'ssp245' or scenario == 'ssp585':
                if parent_model == None:
                    raise Exception('No parent model specified. Need to pass a string for parent model variable, as a future projection is requested.')
                elif parent_model == 'mpi-esm1-2-hr':                    
                    if simulation_type == 'spinup':
                        raise Exception('No isosurface file available.')       
                    elif simulation_type == 'hindcast':
                        if ensemble_run == '000':
                            if temp_resolution == 'monthly':
                                raise Exception('No isosurface file available.')
                            elif temp_resolution == 'daily':
                                model_path =  f"{root_dir}{config}/{scenario}/"                          
                        elif ensemble_run == '001':
                            if temp_resolution == 'monthly':
                                raise Exception('No isosurface file available.')
                            elif temp_resolution == 'daily':
                                model_path =  f"{root_dir}{config}/ensemble_members/{scenario}/ens{ensemble_run}/"                                             
                        else: 
                            raise Exception('No isosurface file available.')       
        return model_path, file_name
    
    @staticmethod
    def open_isosurface_files(model_path,file_name):
        ds_isosurface = xr.open_dataset(model_path+file_name)
        return ds_isosurface

# %%
