#%%
import os
import glob
import sys
import numpy as np
import xarray as xr
import pandas as pd
import scipy.spatial.qhull as qhull
sys.path.append('/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/modules/')
from regridding_tools import Regridder as Regridder


#%%

class ObsGetter():
    """
    author: Eike E. Koehn
    date: June 7, 2024
    description: This class contains a number of functions that enable the loading of observational data. 
    """ 
    @staticmethod
    def get_sst_data(res='monthly', start_year=2011, end_year=2021):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing SST data directly from NCEI NOAA data server
        """
        print(f'Getting {res} SST data.')
        if res == 'daily':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/2d/obs/sst/noaa_oisst_20cm/noaa.oisst.v2.highres/'
            file_urls = []
            years = range(start_year, end_year+1)
            for year in years:    
                file_urls.append(f"{data_path}/sst.day.mean.{year}.nc")
            ds = xr.open_mfdataset(file_urls, parallel=True)
            data_path = 'https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR/'
            file_urls = []
            years = range(start_year, end_year+1)
            for year in years:
                months = range(1, 13)
                for month in months:
                    if month in [1, 3, 5, 7, 8, 10, 12]:
                        days = range(1, 32)
                    elif month in [4, 6, 9, 11]:
                        days = range(1, 31)
                    elif month == 2:
                        if np.mod(year, 4) == 0:
                            days = range(1, 30)
                        else:
                            days = range(1, 29)
                    for day in days:
                        file_urls.append(f"{data_path}{year}{month:02}/oisst-avhrr-v02r01.{year}{month:02}{day:02}.nc")
            ds = xr.open_mfdataset(file_urls)
            
        elif res == 'monthly':
            file_urls = '/nfs/kryo/work/datasets/grd/ocean/2d/obs/sst/noaa_oisst_20cm/noaa.oisst.v2.highres/sst.mon.mean.nc'
            ds = xr.open_dataset(file_urls, engine='netcdf4')
            ds = ds.sel(time=ds.time.dt.year.isin(range(start_year, end_year+1)))

        da = ds.sst
        return ds, da


    @staticmethod
    def get_ssh_data(res='monthly',start_year=2011,end_year=2021):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing SSH data from CMEMS (https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/services) that has been downloaded to ETH's kryo server
        """
        print(f'Getting {res} SSH data.')
        if res == 'daily':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/surface/obs/ssh/duacs_cmems/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D/'
            file_urls = []
            years = range(start_year,end_year+1)
            for year in years:
                months = range(1,13)
                for month in months:
                    if month in [1,3,5,7,8,10,12]:
                        days = range(1,32)
                    elif month in [4,6,9,11]:
                        days = range(1,31)
                    elif month == 2:
                        if np.mod(year,4)==0:
                            days = range(1,30)
                        else:
                            days = range(1,29)
                    for day in days:
                        file_urls.append(glob.glob(f"{data_path}{year}/dt_global_allsat_phy_l4_{year}{month:02}{day:02}_*.nc")[0])
            ds = xr.open_mfdataset(file_urls)
            # rename the dimensions in the dataset
            ds = ds.rename_dims({'latitude':'lat','longitude':'lon'})
            ds = ds.rename_vars({'latitude':'lat','longitude':'lon'})
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
            # 
            da = ds.adt
        elif res == 'monthly':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/surface/obs/ssh/duacs_cmems/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1M-m/'
            file_urls = []
            years = range(start_year,end_year+1)
            for year in years:
                months = range(1,13)
                for month in months:
                    file_urls.append(f"{data_path}{year}/dt_global_allsat_msla_h_y{year}_m{month:02}.nc")
            ds = xr.open_mfdataset(file_urls)
            ds = ds.sel(time=ds.time.dt.year.isin(range(start_year,end_year+1)))
            # rename the dimensions in the dataset
            ds = ds.rename_dims({'latitude':'lat','longitude':'lon'})
            ds = ds.rename_vars({'latitude':'lat','longitude':'lon'})
            # 
            da = ds.sla
        return ds, da

    @staticmethod
    def get_sss_data(res='monthly',start_year=2011,end_year=2021):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing SSS data from CMEMS (https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013/services) that has been downloaded to ETH's kryo server
        """
        print(f'Getting {res} SSS data.')
        if res == 'daily':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/2d/obs/sss/cmems_multiobs/cmems_obs-mob_glo_phy-sss_my_multi_P1D/'
            file_urls = []
            years = range(start_year,end_year+1)
            for year in years:
                months = range(1,13)
                for month in months:
                    if month in [1,3,5,7,8,10,12]:
                        days = range(1,32)
                    elif month in [4,6,9,11]:
                        days = range(1,31)
                    elif month == 2:
                        if np.mod(year,4)==0:
                            days = range(1,30)
                        else:
                            days = range(1,29)
                    for day in days:
                        file_urls.append(glob.glob(f"{data_path}{year}{month:02}/dataset-sss-ssd-rep-daily_{year}{month:02}{day:02}T1200Z_*.nc")[0])
            ds = xr.open_mfdataset(file_urls)
        elif res == 'monthly':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/2d/obs/sss/cmems_multiobs/cmems_obs-mob_glo_phy-sss_my_multi_P1M/'
            file_urls = []
            years = range(start_year,end_year+1)
            for year in years:
                months = range(1,13)
                for month in months:
                    file_urls.append(glob.glob(f"{data_path}{year}/dataset-sss-ssd-rep-monthly_{year}{month:02}*.nc")[0])
            ds = xr.open_mfdataset(file_urls)
            ds = ds.sel(time=ds.time.dt.year.isin(range(start_year,end_year+1)))
        da = ds.sos.sel(depth=0)
        return ds, da

    @staticmethod
    def get_mld_data(res='monthly_clim'):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing MLD data from Holte et al. 2017 (https://mixedlayer.ucsd.edu) that has been downloaded to ETH's kryo server
        """
        print(f'Getting {res} MLD data.')
        if res == 'daily':
            raise Exception('Daily observational MLD data not available.')
        elif res == 'monthly':
            raise Exception('Monthly observational MLD data not available.')
        elif res == 'daily_clim':
            raise Exception('Daily climatological observational MLD data not available.')
        elif res == 'monthly_clim':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/2d/obs/mld/holte/version2018/'
            file_urls = f"{data_path}Argo_mixedlayers_monthlyclim_05092018_mld_dt_mean_convolution_landfill.nc"
            ds = xr.open_dataset(file_urls)
            # mask out the land points
            mask_urls = f"{data_path}Argo_mixedlayers_monthlyclim_05092018_mld_da_mean_convolution.nc"
            ds_mask = xr.open_dataset(mask_urls)
            condition = np.isnan(ds_mask.mld_da_mean)
            ds['mld'] = xr.where(condition,np.NaN,ds.mld)
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
            da = ds.mld 
        return ds, da

    @staticmethod
    def get_stratification_data(res='monthly_clim'):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing upper ocean pycnocline data from Serazin et al. 2022 (https://www.seanoe.org/data/00798/91020/) that has been downloaded to ETH's kryo server.
        """
        print(f'Getting {res} stratification data.')
        if res == 'daily':
            raise Exception('Daily observational stratification data not available.')
        elif res == 'monthly':
            raise Exception('Monthly observational stratification data not available.')
        elif res == 'daily_clim':
            raise Exception('Daily climatological observational stratification data not available.')
        elif res == 'monthly_clim':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/2d/obs/stratification/'
            file_urls = f"{data_path}97791.nc"
            ds = xr.open_dataset(file_urls)
        # rename the dimensions in the dataset
        ds = ds.rename_dims({'latitude':'lat','longitude':'lon'})
        ds = ds.rename_vars({'latitude':'lat','longitude':'lon'})
        #
        da = ds.INTENSITY.sel(stat='mean')
        return ds, da

    @staticmethod
    def get_dic_data(res='monthly_clim'):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing DIC data from /nfs/kryo/work/datasets/grd/ocean/3d/obs/dic/broullon_2020/readme.txt (monthly/annual climatology of DIC from Brullon et al. 2020, website: https://digital.csic.es/handle/10261/200537 )
        """
        print(f'Getting {res} DIC data.')
        if res == 'daily':
            raise Exception('Daily observational data not available.')
        elif res == 'monthly':
            raise Exception('Monthly observational data not available.')
        elif res == 'daily_clim':
            raise Exception('Daily climatological stratification data not available.')
        elif res == 'monthly_clim':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/3d/obs/dic/broullon_2020/'
            file_urls = f"{data_path}TCO2_NNGv2LDEO_climatology.nc"
            ds = xr.open_dataset(file_urls)
            ds = ds.set_coords('depth')
            ds = ds.rename_vars({'depth':'depths'})
            ds = ds.rename_dims({'latitude':'lat','longitude':'lon','time':'month','depth_level':'depth'})
            ds = ds.rename_vars({'latitude':'lat','longitude':'lon','time':'month','depths':'depth'})
            ds = ds.transpose('month','depth', 'lat', 'lon')
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)

            # convert from mumol kg-1 to mmol m-3 by multiplying with a mean density of 1024 kg/m-3, i.e. 1.024 kg/L
            conversion_factor = 1.024 #kg/L
            # -> ideally this should be done with a T,S dependent water density. Could be obtained from WOA2018 T and S. 

            ds['TCO2_NNGv2LDEO'] = ds['TCO2_NNGv2LDEO'] * conversion_factor # convert from mumol/kg; i.e., mumol/kg * kg/L = mumol/L = mmol/m-3
            ds['TCO2_NNGv2LDEO'].attrs["Units"] = 'mmol m-3'

            da = ds.TCO2_NNGv2LDEO
        return ds, da
    
    @staticmethod
    def get_alk_data(res='monthly_clim'):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing Alk data (/nfs/kryo/work/datasets/grd/ocean/3d/obs/alk/broullon_2019/readme.txt) -> - this folder contains the monthly/annual climatology of Total Alkalinity from Brullon et al. 2019
            (Title: A global monthly climatology of total alkalinity: a neural network approach (2019))
            - website: https://digital.csic.es/handle/10261/184460
        """
        print(f'Getting {res} Alk data.')
        if res == 'daily':
            raise Exception('Daily observational data not available.')
        elif res == 'monthly':
            raise Exception('Monthly observational data not available.')
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/3d/obs/alk/broullon_2019/'
            file_urls = f"{data_path}AT_NNGv2_climatology.nc"
            ds = xr.open_dataset(file_urls)
            ds = ds.set_coords('depth')
            ds = ds.rename_vars({'depth':'depths'})
            ds = ds.rename_dims({'latitude':'lat','longitude':'lon','time':'month','depth_level':'depth'})
            ds = ds.rename_vars({'latitude':'lat','longitude':'lon','time':'month','depths':'depth'})
            ds = ds.transpose('month','depth', 'lat', 'lon')
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)

            # convert from mumol kg-1 to mmol m-3 by multiplying with a mean density of 1024 kg/m-3, i.e. 1.024 kg/L
            conversion_factor = 1.024 #kg/L
            # -> ideally this should be done with a T,S dependent water density. Could be obtained from WOA2018 T and S. 

            ds['AT_NNGv2'] = ds['AT_NNGv2'] * conversion_factor # convert from mumol/kg; i.e., mumol/kg * kg/L = mumol/L = mmol/m-3
            ds['AT_NNGv2'].attrs["Units"] = 'mmol m-3'

            da = ds.AT_NNGv2
        return ds, da

    @staticmethod
    def get_pco2_data(res='monthly',start_year=2011,end_year=2021):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing surface pCO2 data https://upwiki.ethz.ch/datasets/gridded/ocean/2d/observation/pco2/ * [[ datasets/gridded/ocean/2d/observation/pco2/oceansoda_gregor//readme.txt | /net/kryo/work/datasets/gridded/ocean/2d/observation/pco2/oceansoda_gregor ]]


        """
        print(f'Getting {res} surface pCO2 data.')
        if res == 'daily':
            raise Exception('Daily observational data not available.')
        elif res == 'monthly':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/2d/obs/pco2/oceansoda_gregor/'
            file_urls = f"{data_path}OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc"
            ds = xr.open_dataset(file_urls)
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
            # make depth a coordinate, not a data variable
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':    
            raise Exception('Monthly climatological data not available.')
        ds = ds.sel(time=ds.time.dt.year.isin(range(start_year,end_year+1)))
        da = ds.spco2
        return ds, da
    
    @staticmethod
    def get_ph_data(res='monthly',start_year=2011,end_year=2021):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing surface pH data 
        """
        print(f'Getting {res} surface pCO2 data.')
        if res == 'daily':
            raise Exception('Daily observational data not available.')
        elif res == 'monthly':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/2d/obs/pco2/oceansoda_gregor/'
            file_urls = f"{data_path}OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc"
            ds = xr.open_dataset(file_urls)
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
            # make depth a coordinate, not a data variable
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':    
            raise Exception('Monthly climatological data not available.')
        ds = ds.sel(time=ds.time.dt.year.isin(range(start_year,end_year+1)))
        da = ds.ph_total
        return ds, da
    
    @staticmethod
    def get_omega_arag_data(res='monthly',start_year=2011,end_year=2021):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing surface omega_aragonite data 
        """
        print(f'Getting {res} omega aragonite data data.')
        if res == 'daily':
            raise Exception('Daily observational data not available.')
        elif res == 'monthly':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/2d/obs/pco2/oceansoda_gregor/'
            file_urls = f"{data_path}OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc"
            ds = xr.open_dataset(file_urls)
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
            ds = ds.sel(time=ds.time.dt.year.isin(range(start_year,end_year+1)))
            da = ds.omega_ar
            # make depth a coordinate, not a data variable
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':    
            data_path = '/nfs/kryo/work/datasets/gridded/ocean/3d/obs/glodap/v2.2016b/'
            file_urls = f"{data_path}GLODAPv2.2016b.OmegaA.nc"
            ds = xr.open_dataset(file_urls)
            ds = ds.rename_dims(depth_surface='depth')
            ds = ds.rename(Depth='depth')
            da = ds.OmegaA
            depth = ds.depth.values
            da = da.assign_coords(depth=("depth", depth))
        return ds, da

    @staticmethod
    def get_temp_data(res='monthly_clim',start_year=2011,end_year=2021):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing temp data 
        """
        print(f'Getting {res} temp data.')
        if res == 'daily':
            raise Exception('Daily observational data not available.')
        elif res == 'monthly':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/3d/obs/en4/EN.4.2.2/' #No such file or directory????
            file_urls = []
            years = range(start_year,end_year+1)
            for year in years:
                months = range(1,13)
                for month in months:
                    file_urls.append(f"{data_path}EN.4.2.2.analyses.c14.{year}/EN.4.2.2.f.analysis.c14.{year}{month:02}.nc")
            ds = xr.open_mfdataset(file_urls)    
            ds['time'] = pd.date_range(f'{start_year}-01-01',f'{end_year}-12-31',freq='1MS')
            da = ds.temperature - 273.15 # Konvert from Kelvin to Celsius 
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':    
            data_path = '/nfs/kryo/work/datasets/grd/ocean/3d/obs/woa/2018/temperature/decav81B0/0.25/'
            file_urls = f"{data_path}woa18_decav81B0_t_monthly.nc"
            ds = xr.open_dataset(file_urls,decode_times=False)
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
            # rename the time dimension
            ds = ds.rename_dims({'time':'month'})
            ds = ds.rename_vars({'time':'month'})
            ds['month'] = np.arange(1,13)
            da = ds.t_an 
        return ds, da

    @staticmethod
    def get_salt_data(res='monthly_clim',start_year=2011,end_year=2021):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing salt data 
        """
        print(f'Getting {res} salt data.')
        if res == 'daily':
            raise Exception('Daily observational data not available.')
        elif res == 'monthly':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/3d/obs/en4/EN.4.2.2/'
            file_urls = []
            years = range(start_year,end_year+1)
            for year in years:
                months = range(1,13)
                for month in months:
                    file_urls.append(f"{data_path}EN.4.2.2.analyses.c14.{year}/EN.4.2.2.f.analysis.c14.{year}{month:02}.nc")
            ds = xr.open_mfdataset(file_urls)    
            ds['time'] = pd.date_range(f'{start_year}-01-01',f'{end_year}-12-31',freq='1MS')
            da = ds.salinity
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':    
            data_path = '/nfs/kryo/work/datasets/grd/ocean/3d/obs/woa/2018/salinity/decav81B0/0.25/'
            file_urls = f"{data_path}woa18_decav81B0_s_monthly.nc"
            ds = xr.open_dataset(file_urls,decode_times=False)
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
            # rename the time dimension
            ds = ds.rename_dims({'time':'month'})
            ds = ds.rename_vars({'time':'month'})
            ds['month'] = np.arange(1,13)
            da = ds.s_an
        return ds, da

    @staticmethod
    def get_no3_data(res='monthly_clim'):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing no3 data 
        """
        print(f'Getting {res} no3 data.')
        if res == 'daily':
            raise Exception('Daily observational data not available.')
        elif res == 'monthly':
            raise Exception('Monthly observational data not available.')
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':    
            data_path = '/nfs/kryo/work/datasets/grd/ocean/3d/obs/woa/2018/nitrate/all/1.00/'
            file_urls = f"{data_path}woa18_all_n_monthly_landfilled_mmolperm3.nc"
            ds = xr.open_dataset(file_urls,decode_times=False)
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
            # rename the time dimension
            ds = ds.rename_dims({'time':'month'})
            ds = ds.rename_vars({'time':'month'})
            ds['month'] = np.arange(1,13)
            # mask out the land points
            mask_urls = f"{data_path}woa18_all_n_monthly.nc"
            ds_mask = xr.open_dataset(mask_urls,decode_times=False)
            ds_mask.coords['lon'] = (ds_mask.coords['lon'] + 360) % 360 #- 180
            ds_mask = ds_mask.sortby(ds_mask.lon)
            # rename the time dimension
            ds_mask = ds_mask.rename_dims({'time':'month'})
            ds_mask = ds_mask.rename_vars({'time':'month'})
            ds_mask['month'] = np.arange(1,13)
            #
            condition = np.isnan(ds_mask.n_an)
            ds['n_an'] = xr.where(condition,np.NaN,ds.n_an)
        da = ds.n_an
        return ds, da

    @staticmethod
    def get_po4_data(res='monthly_clim'):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing po4 data 
        """
        print(f'Getting {res} po4 data.')
        if res == 'daily':
            raise Exception('Daily observational data not available.')
        elif res == 'monthly':
            raise Exception('Monthly observational data not available.')
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':    
            data_path = '/nfs/kryo/work/datasets/grd/ocean/3d/obs/woa/2018/phosphate/all/1.00/'
            file_urls = f"{data_path}woa18_all_p_monthly_landfilled_mmolperm3.nc"
            ds = xr.open_dataset(file_urls,decode_times=False)
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
            # rename the time dimension
            ds = ds.rename_dims({'time':'month'})
            ds = ds.rename_vars({'time':'month'})
            ds['month'] = np.arange(1,13)
            # mask out the land points
            mask_urls = f"{data_path}woa18_all_p_monthly.nc"
            ds_mask = xr.open_dataset(mask_urls,decode_times=False)
            ds_mask.coords['lon'] = (ds_mask.coords['lon'] + 360) % 360 #- 180
            ds_mask = ds_mask.sortby(ds_mask.lon)
            # rename the time dimension
            ds_mask = ds_mask.rename_dims({'time':'month'})
            ds_mask = ds_mask.rename_vars({'time':'month'})
            ds_mask['month'] = np.arange(1,13)
            #
            condition = np.isnan(ds_mask.p_an)
            ds['p_an'] = xr.where(condition,np.NaN,ds.p_an)
        da = ds.p_an
        return ds, da
    
    @staticmethod
    def get_o2_data(res='monthly_clim'):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing o2 data 
        """
        print(f'Getting {res} o2 data.')
        if res == 'daily':
            raise Exception('Daily observational data not available.')
        elif res == 'monthly':
            raise Exception('Monthly observational data not available.')
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':    
            data_path = '/nfs/kryo/work/datasets/grd/ocean/3d/obs/woa/2018/oxygen/all/1.00/'
            file_urls = f"{data_path}woa18_all_o_monthly_landfilled_mmolperm3.nc"
            ds = xr.open_dataset(file_urls,decode_times=False)
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
            # rename the time dimension
            ds = ds.rename_dims({'time':'month'})
            ds = ds.rename_vars({'time':'month'})
            ds['month'] = np.arange(1,13)
            # mask out the land points
            mask_urls = f"{data_path}woa18_all_o_monthly.nc"
            ds_mask = xr.open_dataset(mask_urls,decode_times=False)
            ds_mask.coords['lon'] = (ds_mask.coords['lon'] + 360) % 360 #- 180
            ds_mask = ds_mask.sortby(ds_mask.lon)
            # rename the time dimension
            ds_mask = ds_mask.rename_dims({'time':'month'})
            ds_mask = ds_mask.rename_vars({'time':'month'})
            ds_mask['month'] = np.arange(1,13)
            #
            condition = np.isnan(ds_mask.o_an)
            ds['o_an'] = xr.where(condition,np.NaN,ds.o_an)
        da = ds.o_an
        return ds, da

    @staticmethod
    def get_chl_data(res='monthly',start_year=2011,end_year=2021):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing chl data 
        """
        print(f'Getting {res} chl data.')
        if res == 'daily':
            raise Exception('Daily observational data not available.')
        elif res == 'monthly':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/2d/obs/chl/cmems_globcolour/globcolour_monthly_4km_CHL_REP/'
            file_urls = []
            years = range(start_year,end_year+1)
            for year in years:
                months = range(1,13)
                for month in months:
                    file_urls.append(glob.glob(f"{data_path}{year}/{year}{month:02}01-{year}{month:02d}*_cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M.nc")[0])
            ds = xr.open_mfdataset(file_urls)
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':    
            raise Exception('Monthly observational data not available.')
        da = ds.CHL
        return ds, da

    @staticmethod
    def get_npp_data_preprocessor(dsi):
        dsi = dsi.rename_dims({'fakeDim0':'lat','fakeDim1':'lon'})
        dsi = dsi.expand_dims(dim='time',axis=0)
        dsi.assign_coords({"lon": np.linspace(-180,180,dsi.lon.size), "lat": np.linspace(-90,90,dsi.lat.size)})
        dsi['npp'] = xr.where(dsi.npp<-1000,np.NaN,dsi.npp)
        return dsi
    
    @staticmethod
    def get_npp_data(res='monthly_clim',start_year=2011,end_year=2021):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing npp data 
        """
        print(f'Getting {res} npp data.')
        if res == 'daily':
            raise Exception('Daily observational data not available.')
        elif res == 'monthly':
            data_path = '/nfs/kryo/work/datasets/grd/ocean/2d/obs/npp/vpgm_behrenfeld/nc_files/'
            file_urls = []
            years = range(start_year,end_year+1)
            for year in years:
                monthly_files = sorted(glob.glob(f"{data_path}vgpm.{year}*.nc"))
                for monthly_file in monthly_files:
                    file_urls.append(monthly_file)
            ds = xr.open_mfdataset(file_urls,preprocess=ObsGetter.get_npp_data_preprocessor,concat_dim='time')
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':    
            data_path = '/nfs/kryo/work/datasets/grd/ocean/2d/obs/npp/seawifs/standard_VGPM/'
            file_urls = f"{data_path}vgpm.1998-2007.monthly_clim.nc"
            ds = xr.open_dataset(file_urls,decode_times=False)
            ds = ds.rename_dims({'time':'month','y':'lat','x':'lon'})
            ds = ds.assign_coords({"lon": ds.Lon[ds.Lon[:,0].size//2,:], "lat": ds.Lat[:,ds.Lat[0,:].size//2]})
            ds['month'] = np.arange(1,13)
            ds = ds.drop(['time','Lon','Lat'])
            # move from -180-180 grid to 0-360 longitude grid
            ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 #- 180
            ds = ds.sortby(ds.lon)
        da = ds.npp
        return ds, da
    
    @staticmethod
    def get_cloud_data_preprocessor(dsi, filename):
        #basename = os.path.basename(filename)
        year = int(filename.split('/')[-2])
        doy = int(filename.split('/')[-1][-6:-3])
        date = pd.to_datetime(f'{year}-{doy}', format='%Y-%j')
        dsi = dsi.assign_coords(time=date)#, dim='time')
        for var in dsi.data_vars:
            dsi[var] = dsi[var].expand_dims('time',axis=0)
        return dsi
    
    @staticmethod
    def get_cloud_data(res='daily',start_year=2011,end_year=2021):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing cloud data 
        """
        print(f'Getting {res} cloud data.')
        if res == 'daily':
            data_path = '/nfs/kryo/work/datasets/grd/atm/2d/obs/cloud_fraction/modis_terra/'
            file_urls = []
            years = range(start_year,end_year+1)
            for year in years:
                #if np.mod(year,4)==0:
                #    days = range(1,367)
                #else:
                days = range(1,366)
                for day in days:
                    file_urls.append(f"{data_path}{year}/cloudfraction_day{day:03}.nc")
            ds = xr.open_mfdataset(file_urls,preprocess=lambda ds: ObsGetter.get_cloud_data_preprocessor(ds, ds.encoding['source']))
        elif res == 'monthly':
            raise Exception('Monthly data not available.')
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':    
            raise Exception('Monthly climatological data not available.')
        da = ds.cloud_fraction_sum
        return ds, da

    @staticmethod
    def get_2m_temp_data(res='daily',start_year=2011,end_year=2021):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: accessing 2m temperature data 
        """
        print(f'Getting {res} 2m temp data.')
        if res == 'daily':
            data_path = '/nfs/kryo/work/updata/ecmwf-reanalysis/era5_netcdf/daily/'
            file_urls = []
            years = range(start_year,end_year+1)
            for year in years:
                #if np.mod(year,4)==0:
                #    days = range(1,367)
                #else:
                months = range(1,13)
                for month in months:
                    file_urls.append(f"{data_path}{year}/ERA5_{year}_{month:02}_daily.nc")
            ds = xr.open_mfdataset(file_urls).sel(longitude=slice(200,265),latitude=slice(65,20))
        elif res == 'monthly':
            raise Exception('Monthly data not available.')
        elif res == 'daily_clim':
            raise Exception('Daily climatological data not available.')
        elif res == 'monthly_clim':    
            raise Exception('Monthly climatological data not available.')
        da = ds.t2m - 273.15 # convert to °C
        da = xr.where(np.isnan(ds.sst),da,np.NaN)
        return ds, da

    @staticmethod
    def get_obs_area(ds):
        lon = ds.lon.values
        lat = ds.lat.values
        if len(np.shape(lon))==1 and len(np.shape(lat))==1:
            dlon = np.abs(lon[1]-lon[0])
            dlat = np.abs(lat[1]-lat[0])
            R = 6371000 # radius of earth in m
            longitudinal_circumference = 2*np.pi*R*np.cos(lat*np.pi/180.)
            latitudinal_circumference = 2*np.pi*R
            long_dist = longitudinal_circumference/360*dlon
            lati_dist = latitudinal_circumference/360*dlat
            area = long_dist*lati_dist
            area = np.repeat(area[:,np.newaxis],np.size(lon),axis=1)
        
        obs_area = xr.DataArray(area,coords={'lat':(["lat"],lat),'lon':(["lon"],lon)}, dims=["lat","lon"])
        
        return obs_area

    @staticmethod
    def get_distance_to_coast(model_d2coast,obs_da):
        """
        author: Eike Koehn
        date: Jun 7, 2024
        description: regridding the distance to coast array from ROMS onto the observational lon/lat
        """
        model_lon = model_d2coast.lon.values
        model_lat = model_d2coast.lat.values
        obs_lon = obs_da.lon.values
        obs_lat = obs_da.lat.values
        mod_d2coast = model_d2coast.values
        obs_lon,obs_lat = np.meshgrid(obs_lon,obs_lat)
        d2coast_obs_array = Regridder.regrid_original_to_target(mod_d2coast,model_lon,model_lat,obs_lon,obs_lat)#,target_mask=1)
        d2coast_obs = xr.DataArray(d2coast_obs_array,coords={'lat':(["y","x"],obs_lat),'lon':(["y","x"],obs_lon)}, dims=["y","x"])
        return d2coast_obs

    ####################################################################
    ## NOW THE METHODS THAT CONCERN THE CLIMATOLOGY & EXTREME THRESHOLDS
    ####################################################################

    @staticmethod
    def get_threshold_climatology_dataset(varia,dep,obs_temp_resolution,threshold_type,threshold_value,baseperiod_start_year=2011,baseperiod_end_year=2021,aggregation_kernel=11,smoothing_kernel=31):
        if threshold_type == 'relative':
            if varia == 'temp' and dep == 0 and obs_temp_resolution == 'daily':
                root_dir = "/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/extreme_analysis/thresholds_and_climatology/"
                path_name = '{}oisst/'.format(root_dir) 
                file_name = f'hobday2016_threshold_and_climatology_{varia}_{threshold_value}perc_{baseperiod_start_year}-{baseperiod_end_year}baseperiod_fixedbaseline_{aggregation_kernel}aggregation_{smoothing_kernel}smoothing.nc'
                fn = xr.open_dataset(path_name+file_name)
            elif varia == 'pH_offl' and dep == 0 and obs_temp_resolution == 'daily':
                root_dir = "/nfs/sea/work/fpfaeffli/future_sim/thresholds_and_climatologies/"
                file_name = f'hobday2016_threshold_and_climatology_{varia}_{threshold_value}perc_{baseperiod_start_year}-{baseperiod_end_year}baseperiod_fixedbaseline_{aggregation_kernel}aggregation_{smoothing_kernel}smoothing.nc'
                fn = xr.open_dataset(file_name)
        elif threshold_type == 'absolute':

            raise Exception('Not yet implemented.')
        return fn

    @staticmethod
    def include_feb29(data_365):
        data_365 = data_365.rename({'day_of_year_adjusted':'time'})
        if np.size(data_365.time)==365:
            data_365 = data_365.assign_coords(time=pd.date_range('2001-01-01','2001-12-31'))
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
            data_366['time'] = 1 + np.arange(366)
            data_366 = data_366.rename({'time':'day_of_year_adjusted'})
        return data_366

    @staticmethod
    def get_threshold(variable,depth_level,obs_temp_resolution,threshold_type,threshold_value):#,simulation_type,ensemble_run,temp_resolution,vert_struct,parent_model=None,vtype=None):
        fn = ObsGetter.get_threshold_climatology_dataset(variable,depth_level,obs_temp_resolution,threshold_type,threshold_value)
        if 'depth' in fn.dims:
            threshold = fn.thresh_smoothed.sel(depth=depth_level)#[:,depth_idx,...].values
        else:
            threshold = fn.thresh_smoothed
        threshold_366 = ObsGetter.include_feb29(threshold)
        return threshold, threshold_366
    
    @staticmethod
    def get_climatology(variable,depth_level,obs_temp_resolution,threshold_type,threshold_value):#,simulation_type,ensemble_run,temp_resolution,vert_struct,parent_model=None,vtype=None):
        fn = ObsGetter.get_threshold_climatology_dataset(variable,depth_level,obs_temp_resolution,threshold_type,threshold_value)
        if 'depth' in fn.dims:
            climatology = fn.clim_smoothed.sel(depth=depth_level)
        else:
            climatology = fn.clim_smoothed
        #print(climatology)
        climatology_366 = ObsGetter.include_feb29(climatology)
        return climatology, climatology_366
    
    @staticmethod
    def get_intensity_normalizer(variable,depth_level,obs_temp_resolution,threshold_type,threshold_value):#,simulation_type,ensemble_run,temp_resolution,vert_struct,parent_model=None,vtype=None):
        fn = ObsGetter.get_threshold_climatology_dataset(variable,depth_level,obs_temp_resolution,threshold_type,threshold_value)
        if 'depth' in fn.dims:
            intensity_normalizer = fn.intensity_normalizer_smoothed.sel(depth=depth_level)
        else:
            intensity_normalizer = fn.intensity_normalizer_smoothed
        intensity_normalizer_366 = ObsGetter.include_feb29(intensity_normalizer)
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
            da = da.rename({'day_of_year_adjusted':'time'})
            da = da.assign_coords(time=pd.date_range(f'{year}-01-01',f'{year}-12-31'))
            yearly_files_list.append(da)
        concatenated_data = xr.concat(yearly_files_list,dim='time')
        concatenated_data['time'] = pd.date_range(f'{start_year}-01-01',f'{end_year}-12-31')
        #concatenated_data = concatenated_data.rename({'lat':'eta_rho','lon':'xi_rho'})
        return concatenated_data




# %% WORKING EXAMLPES OF HOW TO LOAD THE DATA
# ds_sst, da_sst = ObsGetter.get_sst_data(res='monthly')
# ds_ssh, da_ssh = ObsGetter.get_ssh_data(res='monthly')
# ds_sss, da_sss = ObsGetter.get_sss_data(res='monthly')
# ds_mld, da_mld = ObsGetter.get_mld_data(res='monthly_clim')
# ds_strat, da_strat = ObsGetter.get_stratification_data(res='monthly_clim')
# ds_dic, da_dic = ObsGetter.get_dic_data(res='monthly_clim')
# ds_alk, da_alk = ObsGetter.get_alk_data(res='monthly_clim')
# ds_pco2, da_pco2 = ObsGetter.get_pco2_data(res='monthly')
# ds_temp, da_temp = ObsGetter.get_temp_data(res='monthly_clim')
# ds_salt, da_salt = ObsGetter.get_salt_data(res='monthly_clim')
# ds_no3,  da_no3  = ObsGetter.get_no3_data(res='monthly_clim')
# ds_po4,  da_po4  = ObsGetter.get_po4_data(res='monthly_clim')
# ds_o2,  da_o2    = ObsGetter.get_o2_data(res='monthly_clim')
# ds_chl, da_chl = ObsGetter.get_chl_data(res='monthly')
# ds_npp, da_npp = ObsGetter.get_npp_data(res='monthly_clim')
# ds_cloud, da_cloud = ObsGetter.get_cloud_data(res='daily')

# %%

import xarray as xr

# Open your NetCDF file
ds = xr.open_dataset('/nfs/kryo/work/datasets/grd/ocean/2d/obs/pco2/oceansoda_gregor/OceanSODA_ETHZ-v2023.OCADS.01_1982-2022.nc')

# Extract the `ph_total` and `ph_free` variables
ph_total_data = ds['ph_total']
ph_free_data = ds['ph_free']

# Display basic information about the data
print(ph_total_data)
print(ph_free_data)

# Print a small portion of the `ph_total` values
print(ph_total_data.values)  

# Alternatively, select a small slice for inspection
print(ph_total_data.isel(time=0, lat=slice(0, 5), lon=slice(0, 5)).values)



# %%
