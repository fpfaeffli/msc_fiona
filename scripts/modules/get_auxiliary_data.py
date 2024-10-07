#%%
import numpy as np
import xarray as xr

#%%
class AuxGetter():
    """
    author: Eike E. Koehn
    date: June 7, 2024
    description: This class contains a number of functions that enable the loading of auxiliary data. 
    The functions include:
    1. getting etopo data to plot the landmask
    2. plotting the landmask based on etopo data
    """ 

    def get_etopo_data():
        # load the ETOPO data (Earth Topography data) -> function uses 
        # ETOPO data to identify where land and water are located
        topopath='/nfs/kryo/work/updata/bathymetry/ETOPO1/'
        topofile='ETOPO5_mean_bath.nc'
        ds = xr.open_dataset(topopath+topofile)
        ds = ds.rename_dims({'latitude':'lat','longitude':'lon'})
        ds = ds.rename_vars({'latitude':'lat','longitude':'lon'})
        # shuffle coordinates
        ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
        ds = ds.sortby(ds.lon)
        # generate land mask
        da = xr.where(ds.mean_bath>0,1,np.NaN)
        return da

# %%
