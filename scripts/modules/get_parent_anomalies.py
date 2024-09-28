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

class ParentGetter():
    """
    author: Eike E. Koehn
    date: June 7, 2024
    description: This class contains a number of functions that enable the loading of simulation data. 
    """ 

    @staticmethod
    def define_parent_setups():
        """
        author: Eike E. Koehn
        date: June 10, 2024
        description: Specify the different parent model setups.
        """
        scenarios = ['ssp245','ssp585']
        parent_models = ['mpi-esm1-2-hr']
        vtypes = ['oceanic','atmospheric']
        return scenarios, parent_models, vtypes
    
    ##########################################################################
    ## NOW THE METHODS THAT ALLOW THE RETRIEVAL OF THE PARENT MODEL DELTAS ###
    ##########################################################################

    @staticmethod
    def get_parent_anomaly_paths(scenario,vtype,parent_model=None):
        """
        author: Eike E. Koehn
        date: June 10, 2024
        description: Specify the different parent model anomaly paths.
        """        

        root_dir = "/net/kryo/work/geirund/climate/deltas/"

        if scenario == 'ssp245':
            model_path = root_dir + 'SSP245/'
        elif scenario == 'ssp585':
            model_path = root_dir + 'SSP585/'

        if vtype == 'oceanic':
            model_path += 'Omon/'
        elif vtype == 'atmospheric':
            model_path += 'Amon/'

        if parent_model == 'mpi-esm1-2-hr':
            model_path += 'MPI-ESM1-2-HR/'
        elif parent_model != 'mpi-esm1-2-hr':
            raise Exception('Only MPI-ESM1-2-HR used so far as parent model.')
        
        model_path += 'interpolated_to_roms_grid/'
        
        return model_path
    
    @staticmethod
    def open_parent_anomaly_datasets(parent_path,var,dep):
        """
        author: Eike E. Koehn
        date: June 10, 2024
        description: Open the parent anomaly dataset
        """        

        if var == 'temp' and dep != 0:
            model_filename = 'thetao_delta_landfill.nc'
            varname = 'thetao'
        elif var == 'temp' and dep == 0:
            model_filename = 'tos_delta.nc'
            varname = 'tos'
        else:
            raise Exception('For this variable no associated anomaly file is defined as of now. Check in function open_parent_anomaly_datasets() in get_parent_anomalies.py file.')
        
        ds = xr.open_dataset(parent_path+model_filename)
        da = ds[varname]

        return ds,da