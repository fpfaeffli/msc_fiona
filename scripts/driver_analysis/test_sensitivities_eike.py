#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Adapted for numpy/ma/cdms2 by convertcdms.py
# !/usr/bin/env python                                                    
# Adapted from CMIP5_mocsy.py example from gitHub                                                                     
#                                                                         
# <*>=                                                                    
# <Module Documentation>=   


### This code takes 39sec to run for each day thus 4hours for a year.
### Thus I would need : 41 * 4 hours = 164 hours = 7 days to re run the carbonate chemistry. 

### To run this script, activate the conda environment mocsy_fdesmet (conda activate /home/fdesmet/.conda/envs/mocsy_fdesmet)

"""
    From modelled DIC and Alkalinity, Temperature and Salinity, and Silica and Phosphate:
    compute the following variables:
        - pH (Total scale),
        - ocean partial pressure of CO2 (uatm),
        - CO2 fugacity (uatm),
        - CO2*, HCO3- and CO3-- concentrations (mol/m^3),
        - Omega-C and Omega-A, saturation state of calcite and aragonite,
        - BetaD, homogeneous buffer factor (a.k.a. the Revelle Factor)
    Save them in a NetCDF file.

    This python script calls a Fortran subroutine (mocsy), with recent developments by J.C. Orr.
    It was derived from original code used by O. Aumont for carb. chem. in OPA-PISCES model,
    part of which was based on code from E. Maier-Reimer (HAMMOC3); other parts were taken from OCMIP2.
    That code was first extended and corrected by J.-M. Epitalon & J. Orr, and also 
    made consistent with J-P Gattuso's Seacarb R software (in 2004). This effort provided bug
    fixes to seacarb (sent to J.P. Gattuso who implemented them in the
    subsequent seacarb version).  Originally, this python-fortran code was
    used for the OCMIP-2 & GLODAP analysis described in (Orr et al., 2005,
    Nature). It has subsequently undergone improvements and functionalities added
    by J. Orr.  Extensive tests 2012 and 2013 show it produces
    results that are essentially identical to those from CO2sys and seacarb 
    (within round-off error).
"""

#%%

import numpy as np
import xarray as xr
import sys, os
#mocsy_dir = "/home/fdesmet/Roms/new_mocsy_2_0/mocsy/"
#sys.path.append(mocsy_dir)  
#import mocsy
import glob
import pandas as pd
import matplotlib.pyplot as plt
# load in the functions to calculate the climatology
sys.path.append('/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/climatologies_and_thresholds/')
from funcs_for_clim_thresh import * 
import dask

grid_zlevs = np.array([0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60, -65, -70, -75, -80, -85, -90, -95, -100, -110, -120, -130, -140, -150, -165, -180, -200, -225, -250, -280, -310, -350, -400, -450, -500])

#print('GET THE CASE')
#exec(open('../modules/define_cases_and_parameters.py').read())
#casedir = '/home/koehne/Documents/publications/paper_future_simulations/scripts/modules/cases/'   # /home/koehne/...
#casename = 'case00.yaml'
#params = read_config_files(casedir+casename)

#%%
eta_rho_cho = 500
xi_rho_cho = 200
depthlevel = 0
year = 2015
month = 'all' #6 # 'all'
scenario = 'present'

if month == 'all':
    ds = xr.open_mfdataset(sorted(glob.glob(f'/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/romsoc_fully_coupled/{scenario}/daily/z_avg/z_avg_2015_*_37zlevs_full_1x1meanpool_downsampling.nc')),concat_dim='time',combine='nested')
    ds['time'] = pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D')

    ds_sens = xr.open_mfdataset(sorted(glob.glob(f'/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/romsoc_fully_coupled/{scenario}/daily/z_avg/mocsy_co2_chemistry/{year}/pactcs30_romsoc_fully_coupled_{scenario}_daily_{year}_*_mocsy_sensitivities_depthlevel_0.nc')))
else:
    ds = xr.open_dataset(f'/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/romsoc_fully_coupled/{scenario}/daily/z_avg/z_avg_2015_{month:03d}_37zlevs_full_1x1meanpool_downsampling.nc')
    #ds['time'] = pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D')

    ds_sens = xr.open_dataset(f'/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/romsoc_fully_coupled/{scenario}/daily/z_avg/mocsy_co2_chemistry/{year}/pactcs30_romsoc_fully_coupled_{scenario}_daily_{year}_{month:03d}_mocsy_sensitivities_depthlevel_0.nc')   

#%%
omegaa = ds.omega_arag_offl.isel(depth=depthlevel,eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
pH = ds.pH_offl.isel(depth=depthlevel,eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
hplus = 10**(-pH)
temp = ds.temp.isel(depth=depthlevel,eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
dic = ds.DIC.isel(depth=depthlevel,eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
alk = ds.Alk.isel(depth=depthlevel,eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
salt = ds.salt.isel(depth=depthlevel,eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)

dh_dtemp = ds_sens.dh_dtem.isel(eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
dh_dsalt = ds_sens.dh_dsal.isel(eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
dh_ddic = ds_sens.dh_ddic.isel(eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
dh_dalk = ds_sens.dh_dalk.isel(eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)

domegaa_dtemp = ds_sens.domegaa_dtem.isel(eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
domegaa_dsalt = ds_sens.domegaa_dsal.isel(eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
domegaa_ddic = ds_sens.domegaa_ddic.isel(eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
domegaa_dalk = ds_sens.domegaa_dalk.isel(eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)

#%% Plot an individual timeseries throughout a full year with the anomalies relative to the annual mean

fig, ax = plt.subplots(2,1,figsize=(10,8))
ax[0].plot(hplus-np.mean(hplus),label=r'Model output $H^{+}_{anom}$',color='k',linewidth=3)
ax[0].plot(dh_dtemp*(temp-np.mean(temp)),label=r'$\frac{\partial H^{+}}{\partial T} T_{anom}$',color='C1',linewidth=2)
ax[0].plot(dh_dsalt*(salt-np.mean(salt)),label=r'$\frac{\partial H^{+}}{\partial S} S_{anom}$',color='C3',linewidth=2)
ax[0].plot(dh_ddic*(dic-np.mean(dic)),label=r'$\frac{\partial H^{+}}{\partial DIC} DIC_{anom}$',color='C0',linewidth=2)
ax[0].plot(dh_dalk*(alk-np.mean(alk)),label=r'$\frac{\partial H^{+}}{\partial Alk} Alk_{anom}$',color='C2',linewidth=2)
ax[0].plot(
    dh_dtemp*(temp-np.mean(temp)) +
    dh_dsalt*(salt-np.mean(salt)) +
    dh_ddic*(dic-np.mean(dic)) +
    dh_dalk*(alk-np.mean(alk)),
    label='sum of linearized terms',linewidth=3,color='C4')
ax[0].legend(loc='lower left',bbox_to_anchor=(1.02,0.1))

ax[1].plot(omegaa-np.mean(omegaa),label=r'Model output $\Omega_{anom}$',color='k',linewidth=3)
ax[1].plot(domegaa_dtemp*(temp-np.mean(temp)),label=r'$\frac{\partial \Omega_{A}}{\partial T} T_{anom}$',color='C1',linewidth=2)
ax[1].plot(domegaa_dsalt*(salt-np.mean(salt)),label=r'$\frac{\partial\Omega_{A}}{\partial S} S_{anom}$',color='C3',linewidth=2)
ax[1].plot(domegaa_ddic*(dic-np.mean(dic)),label=r'$\frac{\partial \Omega_{A}}{\partial DIC} DIC_{anom}$',color='C0',linewidth=2)
ax[1].plot(domegaa_dalk*(alk-np.mean(alk)),label=r'$\frac{\partial \Omega_{A}}{\partial Alk} Alk_{anom}$',color='C2',linewidth=2)
ax[1].plot(
    domegaa_dtemp*(temp-np.mean(temp)) +
    domegaa_dsalt*(salt-np.mean(salt)) +
    domegaa_ddic*(dic-np.mean(dic)) +
    domegaa_dalk*(alk-np.mean(alk)),
    label='sum of linearized terms',linewidth=3,color='C4')
ax[1].legend(loc='lower left',bbox_to_anchor=(1.02,0.1))

# %% Now plot the same thing but with the anomalies not relative to the annual mean, but relative to the climatologies of H+, Omega, T, S, DIC, Alk
clim_dir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/extreme_analysis/thresholds_and_climatology/romsoc_fully_coupled/present/'

#%% get h clim
ds_hclim = xr.open_dataset(clim_dir+'hobday2016_threshold_and_climatology_Hions_37zlevs_full_1x1meanpool_downsampling_2011-2021analysisperiod_90perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing.nc')
hplus_clim = ds_hclim.clim_smoothed.isel(depth=depthlevel,lat = eta_rho_cho, lon=xi_rho_cho)
hplus_clim['time'] = pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D')

#%% get omega clim
ds_omegaclim = xr.open_dataset(clim_dir+'hobday2016_threshold_and_climatology_omega_arag_offl_37zlevs_full_1x1meanpool_downsampling_2011-2021analysisperiod_10perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing.nc')
omega_clim = ds_omegaclim.clim_smoothed.isel(depth=depthlevel,lat = eta_rho_cho, lon=xi_rho_cho)
omega_clim['time'] = pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D')

#%% get t clim
ds_tclim = xr.open_dataset(clim_dir+'hobday2016_threshold_and_climatology_temp_37zlevs_full_1x1meanpool_downsampling_2011-2021analysisperiod_90perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing.nc')
t_clim = ds_tclim.clim_smoothed.isel(depth=depthlevel, lat = eta_rho_cho, lon=xi_rho_cho)
t_clim['time'] = pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D')

#%% load in full s, dic, alk for climatology calculation (because the climatologies have not yet been calculated) - will calculate them here on the fly even though it takes a while
ds_all = xr.open_mfdataset(sorted(glob.glob(f'/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/romsoc_fully_coupled/{scenario}/daily/z_avg/z_avg_*_37zlevs_full_1x1meanpool_downsampling.nc')),concat_dim='time',combine='nested',parallel=True)
ds_all['time'] = pd.date_range(f'2010-01-01',f'2021-12-31',freq='D')
ds_all = ds_all.sel(time=slice('2011-01-01','2021-12-31'))

#%% get s clim
ds_s = ds_all.salt.isel(depth=depthlevel,eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
da_without_leap_day = ThreshClimFuncs.drop_29th_feb(ds_s)
climatology = da_without_leap_day.rolling(time=11, min_periods=1, center=True).construct('window_dim').groupby('day_of_year_adjusted').apply(ThreshClimFuncs.calculate_mean)
kernel = np.ones(31)
dim = 'day_of_year_adjusted'
axis = climatology.dims.index(dim)
s_clim = xr.apply_ufunc(ThreshClimFuncs.convolve_along_axis, climatology, kernel, kwargs={'axis': axis},dask='allowed')#, input_core_dims=[[dim]], output_core_dims=[[dim]], vectorize=True)      
s_clim = s_clim.rename({'day_of_year_adjusted':'time'})
s_clim['time'] = pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D')

#%% get dic clim
ds_dic = ds_all.DIC.isel(depth=depthlevel,eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
da_without_leap_day = ThreshClimFuncs.drop_29th_feb(ds_dic)
climatology = da_without_leap_day.rolling(time=11, min_periods=1, center=True).construct('window_dim').groupby('day_of_year_adjusted').apply(ThreshClimFuncs.calculate_mean)
kernel = np.ones(31)
dim = 'day_of_year_adjusted'
axis = climatology.dims.index(dim)
dic_clim = xr.apply_ufunc(ThreshClimFuncs.convolve_along_axis, climatology, kernel, kwargs={'axis': axis},dask='allowed')#, input_core_dims=[[dim]], output_core_dims=[[dim]], vectorize=True)    
dic_clim = dic_clim.rename({'day_of_year_adjusted':'time'})
dic_clim['time'] = pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D')

#%% get alk clim
ds_alk = ds_all.Alk.isel(depth=depthlevel,eta_rho = eta_rho_cho, xi_rho=xi_rho_cho)
da_without_leap_day = ThreshClimFuncs.drop_29th_feb(ds_alk)
climatology = da_without_leap_day.rolling(time=11, min_periods=1, center=True).construct('window_dim').groupby('day_of_year_adjusted').apply(ThreshClimFuncs.calculate_mean)
kernel = np.ones(31)
dim = 'day_of_year_adjusted'
axis = climatology.dims.index(dim)
alk_clim = xr.apply_ufunc(ThreshClimFuncs.convolve_along_axis, climatology, kernel, kwargs={'axis': axis},dask='allowed')#, input_core_dims=[[dim]], output_core_dims=[[dim]], vectorize=True)   
alk_clim = alk_clim.rename({'day_of_year_adjusted':'time'})
alk_clim['time'] = pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D')

# %% Calculate the anomalies relative to the climatology

h_anom = hplus - hplus_clim
omegaa_anom = omegaa - omega_clim
t_anom = temp - t_clim
s_anom = salt - s_clim
dic_anom = dic - dic_clim
alk_anom = alk - alk_clim

#%% Plot an individual timeseries throughout a full year with the anomalies relative to the annual mean

fig, ax = plt.subplots(2,1,figsize=(15,11),sharex=True)
ax[0].plot(10**9*h_anom,label=r'Model output $H^{+}_{anom}$',color='k',linewidth=3)
ax[0].plot(10**9*dh_dtemp*t_anom,label=r'$\frac{\partial H^{+}}{\partial T} T_{anom}$',color='C1',linewidth=2)
ax[0].plot(10**9*dh_dsalt*s_anom,label=r'$\frac{\partial H^{+}}{\partial S} S_{anom}$',color='C3',linewidth=2)
ax[0].plot(10**9*dh_ddic*dic_anom,label=r'$\frac{\partial H^{+}}{\partial DIC} DIC_{anom}$',color='C0',linewidth=2)
ax[0].plot(10**9*dh_dalk*alk_anom,label=r'$\frac{\partial H^{+}}{\partial Alk} Alk_{anom}$',color='C2',linewidth=2)
ax[0].plot(10**9*
    (dh_dtemp*t_anom +
    dh_dsalt*s_anom +
    dh_ddic*dic_anom +
    dh_dalk*alk_anom),
    label='sum of linearized terms',linewidth=3,color='C4')
ax[0].legend(loc='lower left',bbox_to_anchor=(1.02,0.1))
ax[0].set_ylabel('H+ ions anom. (rel to clim.) in nmol L')

ax[1].plot(omegaa_anom,label=r'Model output $\Omega_{anom}$',color='k',linewidth=3)
ax[1].plot(domegaa_dtemp*t_anom,label=r'$\frac{\partial \Omega_{A}}{\partial T} T_{anom}$',color='C1',linewidth=2)
ax[1].plot(domegaa_dsalt*s_anom,label=r'$\frac{\partial\Omega_{A}}{\partial S} S_{anom}$',color='C3',linewidth=2)
ax[1].plot(domegaa_ddic*dic_anom,label=r'$\frac{\partial \Omega_{A}}{\partial DIC} DIC_{anom}$',color='C0',linewidth=2)
ax[1].plot(domegaa_dalk*alk_anom,label=r'$\frac{\partial \Omega_{A}}{\partial Alk} Alk_{anom}$',color='C2',linewidth=2)
ax[1].plot(
    domegaa_dtemp*t_anom +
    domegaa_dsalt*s_anom +
    domegaa_ddic*dic_anom +
    domegaa_dalk*alk_anom,
    label='sum of linearized terms',linewidth=3,color='C4')
ax[1].legend(loc='lower left',bbox_to_anchor=(1.02,0.1))
ax[1].set_ylabel('Omega Aragonite anom. (rel to clim.)')
ax[1].set_xlabel(f'Day of year {year}')
plt.tight_layout()
plt.savefig('decomposition_hplus_anom_rel_to_clim.png',dpi=200)

# %%
fig,ax = plt.subplots(3,2,figsize=(11,11),sharex=True)
ax[0,0].plot(10**9*hplus,color='k',label='H+')
ax[0,0].plot(10**9*hplus_clim,color='k',linestyle='--',label='H+ clim')
ax[0,0].set_ylabel('H+ ions in nmol L')
ax[0,1].plot(omegaa,color='k',label='Omega')
ax[0,1].plot(omega_clim,color='k',linestyle='--',label='Omega clim')
ax[0,1].set_ylabel('Omega Arag.')
ax[1,0].plot(temp,color='C1',label='SST')
ax[1,0].plot(t_clim,color='C1',linestyle='--',label='SST clim')
ax[1,0].set_ylabel('Temp. in °C')
ax[1,1].plot(salt,color='C3',label='SSS')
ax[1,1].plot(s_clim,color='C3',linestyle='--',label='SSS clim')
ax[1,1].set_ylabel('Salinity')
ax[2,0].plot(dic,color='C0',label='SSDIC')
ax[2,0].plot(dic_clim,color='C0',linestyle='--',label='DIC clim')
ax[2,0].set_ylabel('DIC in mmol m-3')
ax[2,1].plot(alk,color='C2',label='SSAlk')
ax[2,1].plot(alk_clim,color='C2',linestyle='--',label='Alk clim')
ax[2,1].set_ylabel('Alkalinity in mmol m-3')
for axi in ax[-1,:]:
    axi.set_xlabel(f'Day of year {year}')
for axi in ax.flatten():
    axi.legend()
plt.tight_layout()
plt.savefig('decomposition_associated_timeseries.png',dpi=200)


#%% Plot an individual timeseries throughout a full year with the anomalies relative to previous time step

fig, ax = plt.subplots(2,1,figsize=(15,11),sharex=True)
ax[0].plot(10**9*(hplus.diff(dim='time',n=1)),label=r'Model output $\Delta H^{+}$',color='k',linewidth=3.5)
ax[0].plot(10**9*dh_dtemp.isel(time=slice(0,-1))*temp.diff(dim='time',n=1),label=r'$\frac{\partial H^{+}}{\partial T} \Delta T$',color='C1',linewidth=1)
ax[0].plot(10**9*dh_dsalt.isel(time=slice(0,-1))*salt.diff(dim='time',n=1),label=r'$\frac{\partial H^{+}}{\partial S} \Delta S$',color='C3',linewidth=1)
ax[0].plot(10**9*dh_ddic.isel(time=slice(0,-1))*dic.diff(dim='time',n=1),label=r'$\frac{\partial H^{+}}{\partial DIC} \Delta DIC$',color='C0',linewidth=1)
ax[0].plot(10**9*dh_dalk.isel(time=slice(0,-1))*alk.diff(dim='time',n=1),label=r'$\frac{\partial H^{+}}{\partial Alk} \Delta Alk$',color='C2',linewidth=1)

ax[0].plot(10**9*
    (dh_dtemp.isel(time=slice(0,-1))*temp.diff(dim='time',n=1) +
    dh_dsalt.isel(time=slice(0,-1))*salt.diff(dim='time',n=1) +
    dh_ddic.isel(time=slice(0,-1))*dic.diff(dim='time',n=1) +
    dh_dalk.isel(time=slice(0,-1))*alk.diff(dim='time',n=1)),
    label='sum of linearized terms',linewidth=2,color='C4')
ax[0].legend(loc='lower left',bbox_to_anchor=(1.02,0.1))
ax[0].set_ylabel('$\Delta H+$ ions (rel to. prev. time step) in nmol L')


ax[1].plot(10**9*(omegaa.diff(dim='time',n=1)),label=r'Model output $\Delta \Omega_A$',color='k',linewidth=3.5)
ax[1].plot(10**9*domegaa_dtemp.isel(time=slice(0,-1))*temp.diff(dim='time',n=1),label=r'$\frac{\partial \Omega_A}{\partial T} \Delta T$',color='C1',linewidth=1)
ax[1].plot(10**9*domegaa_dsalt.isel(time=slice(0,-1))*salt.diff(dim='time',n=1),label=r'$\frac{\partial \Omega_A}{\partial S} \Delta S$',color='C3',linewidth=1)
ax[1].plot(10**9*domegaa_ddic.isel(time=slice(0,-1))*dic.diff(dim='time',n=1),label=r'$\frac{\partial \Omega_A}{\partial DIC} \Delta DIC$',color='C0',linewidth=1)
ax[1].plot(10**9*domegaa_dalk.isel(time=slice(0,-1))*alk.diff(dim='time',n=1),label=r'$\frac{\partial \Omega_A}{\partial Alk} \Delta Alk$',color='C2',linewidth=1)

ax[1].plot(10**9*
    (domegaa_dtemp.isel(time=slice(0,-1))*temp.diff(dim='time',n=1) +
    domegaa_dsalt.isel(time=slice(0,-1))*salt.diff(dim='time',n=1) +
    domegaa_ddic.isel(time=slice(0,-1))*dic.diff(dim='time',n=1) +
    domegaa_dalk.isel(time=slice(0,-1))*alk.diff(dim='time',n=1)),
    label='sum of linearized terms',linewidth=2,color='C4')
ax[1].legend(loc='lower left',bbox_to_anchor=(1.02,0.1))
ax[1].set_ylabel('$\Delta \Omega_A$ (rel to. prev. time step)')

ax[1].set_xlabel(f'Day of year {year}')

for axi in ax:
    axi.grid()
plt.tight_layout()
plt.savefig('decomposition_hplus_omega_delta.png',dpi=200)

# %%
