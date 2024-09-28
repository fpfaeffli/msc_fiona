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
class CarbonateChemistryGetter():
    """
    author: Eike E. Koehn
    date: June 7, 2024
    description: This class contains a number of functions that enable the loading of carbonate chemistry variables. 
    """ 
    
    @staticmethod
    def get_isosurface_files(config,scenario,simulation_type,ensemble_run,temp_resolution,parent_model=None):
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

