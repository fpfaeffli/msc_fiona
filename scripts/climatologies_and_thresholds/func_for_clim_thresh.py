"""
author: Eike E. Köhn
date: June 21, 2024
description: This file contains a collection of functions that are used in the calculation of the thresholds and climatologies, for both models and observations.
"""

#%%#
import numpy as np
import scipy.ndimage as spndimage
import xarray as xr
import pandas as pd

#%% DEFINE CORE FUNCTIONS

class ThreshClimFuncs:

    # Step 2: Group by day of year and calculate percentile
    def calculate_percentile(dd,q):
        return dd.quantile(q, dim=('time','window_dim'))
    
    # Function to calculate mean
    def calculate_mean(dd):
        return dd.mean(dim=('time','window_dim'))
    
    # derop the 29th of february
    def drop_29th_feb(ds): 
        # get the day of year vector
        day_of_year = ds.time.dt.dayofyear
        is_leap_year = ds.time.dt.is_leap_year
        # Select February 29th entries
        is_leap_day = (ds['time.month'] == 2) & (ds['time.day'] == 29)
        # Invert the mask to get all non-leap day entries
        not_leap_day = ~is_leap_day
        # Apply the mask to dataset, i.e. remove all leap days
        ds_without_leap_day = ds.sel(time=not_leap_day)
        # now adjust the day of year vector and include it as coordinate
        day_of_year_adjusted = day_of_year*1.
        day_of_year_adjusted[is_leap_year & (day_of_year >= 60)] -= 1
        ds_without_leap_day['day_of_year_adjusted'] = day_of_year_adjusted
        ds_without_leap_day.coords['day_of_year_adjusted'] = ds_without_leap_day['day_of_year_adjusted']
        return ds_without_leap_day

    def calc_thresh(params,da):
        # Perform rolling window operation
        da_without_leap_day = ThreshClimFuncs.drop_29th_feb(da)
        threshold = da_without_leap_day.rolling(time=params.aggregation_window_size, min_periods=1, center=True).construct('window_dim').groupby('day_of_year_adjusted').map(ThreshClimFuncs.calculate_percentile,q=params.percentile/100)
        return threshold
    
    def calc_clim(params,da):
        da_without_leap_day = ThreshClimFuncs.drop_29th_feb(da)
        climatology = da_without_leap_day.rolling(time=params.aggregation_window_size, min_periods=1, center=True).construct('window_dim').groupby('day_of_year_adjusted').apply(ThreshClimFuncs.calculate_mean)
        return climatology
    
    def calc_intensity_normalizer(threshold,climatology):
        return threshold-climatology

    # Define a function to apply convolve1d along a specific axis
    def convolve_along_axis(arr, kernel, axis):
        return spndimage.convolve1d(arr, kernel, axis=axis, mode='wrap')/np.size(kernel)

    # Apply the convolve1d function along the 'day_of_year_adjusted' axis of the DataArray
    def smooth_array(params,da):
        kernel = np.ones(params.smoothing_window_size)
        dim = 'day_of_year_adjusted'
        axis = da.dims.index(dim)
        da_smoothed = xr.apply_ufunc(ThreshClimFuncs.convolve_along_axis, da, kernel, kwargs={'axis': axis})#, input_core_dims=[[dim]], output_core_dims=[[dim]], vectorize=True)      
        return da_smoothed
    
    def put_fields_into_dataset(params,climatology_smoothed,threshold_smoothed,intensity_normalizer_smoothed,obs_ds):
        out_ds = xr.Dataset()
        out_ds['clim_smoothed'] = climatology_smoothed
        out_ds['thresh_smoothed'] = threshold_smoothed
        out_ds['intensity_normalizer_smoothed'] = intensity_normalizer_smoothed
        # add some attributes
        out_ds.attrs['case_description'] = 'Given by the following attributes:'
        for pattribute in dir(params):
            if not pattribute.startswith('__') and not callable(getattr(params,pattribute)):
                if pattribute != 'param_names':
                    print(pattribute)
                    out_ds.attrs['params.'+pattribute] = eval('params.'+pattribute)
        #out_ds.attrs['data_source_file'] = obs_ds['time'].encoding['source']
        out_ds.attrs['leap_years'] = 'Feb 29th values were discarded for the computation of the threshold and climatologies. They are thus 365 days long. In order to get a 366th value for leap years, I propose to linearly interpolate the values between Feb 28 and Mar 1.'
        return out_ds
    
    def repeat_array_multiple_years(array,start_year=2011,end_year=2021):
        # compute a 29th of february value and extend the array by that value, so that it features 366 days.
        feb29 = (array.sel(time='2015-02-28') + array.sel(time='2015-03-01'))/2
        feb29 = feb29.assign_coords({'time':pd.to_datetime('2016-02-29')})
        part1_365 = array.sel(time=slice('2015-01-01','2015-02-28'))
        part1_365 = part1_365.assign_coords({'time':pd.date_range(f'2016-01-01',f'2016-02-28')})
        part3_365 = array.sel(time=slice('2015-03-01','2015-12-31'))
        part3_365 = part3_365.assign_coords({'time':pd.date_range(f'2016-03-01',f'2016-12-31')})     
        # print(feb29)
        # print(part1_365)
        # print(part3_365)   
        array_366 = xr.concat((part1_365,feb29,part3_365),dim='time')

        year_list = np.arange(start_year,end_year+1)
        to_concatenate = []
        for year in year_list:
            if np.mod(year,4)==0:
                array_366 = array_366.assign_coords({'time':pd.date_range(f'{year}-01-01',f'{year}-12-31')})
                to_concatenate.append(array_366)
            else:
                array = array.assign_coords({'time':pd.date_range(f'{year}-01-01',f'{year}-12-31')})
                to_concatenate.append(array)
        to_concatenate = tuple(to_concatenate)
        repeated_array = xr.concat(to_concatenate,dim='time')
        return repeated_array
    
    # def save_to_netcdf(params,var,out_ds):
    #     print("Saving")
    #     # set the path and filename
    #     savepath = params.rootdir + 'oisst/'
    #     save_filename = f'hobday2016_threshold_and_climatology_{var}_{params.percentile}perc_{params.baseline_start_year}-{params.baseline_end_year}baseperiod_{params.baseline_type}baseline_{params.aggregation_window_size}aggregation_{params.smoothing_window_size}smoothing.nc'
    #     # save
    #     out_ds.to_netcdf(savepath+save_filename)
    #     print("Saving done")
