#%%
import os
import glob
import sys
import numpy as np
import xarray as xr
import scipy.spatial.qhull as qhull


#%%

class Regridder():
    """
    author: Eike Koehn
    date: Jun 14, 2024
    description: This regridder work only for a 2d dataset. These functions work adapted from M. Frischknecht for my needs.
    """
    @staticmethod
    def regrid_original_to_target(original_o,lon_o,lat_o,target_lon,target_lat):#,target_mask=1):
        """Function to regrid data (original_o) from an original grid (lon_o,lat_o) to a target grid."""

        if len(np.shape(target_lat))==2:
            targSY = np.shape(target_lat)[0]
            targSX = np.shape(target_lat)[1]
        else:
            target_lon,target_lat = np.meshgrid(target_lon,target_lat)
            targSY = np.shape(target_lat)[0]
            targSX = np.shape(target_lat)[1]

        if np.size(np.shape(lon_o))==1 and np.size(np.shape(lat_o))==1:
            lonORIG = lonORIG,latORIG = np.meshgrid(lon_o,lat_o)
        else:
            lonORIG = lon_o
            latORIG = lat_o
        lonORIG_flat = lonORIG.flatten()
        latORIG_flat = latORIG.flatten()
        xy = np.zeros([len(latORIG_flat),2])
        xy[:,0] = lonORIG_flat
        xy[:,1] = latORIG_flat
        
        # ROMS Lon Lat
        lonTARG_flat = target_lon.flatten()
        latTARG_flat = target_lat.flatten()
        xyTARG = np.zeros([len(latTARG_flat),2])
        xyTARG[:,0] = lonTARG_flat
        xyTARG[:,1] = latTARG_flat

        # Compute triangulation of grid points
        tri = Regridder.interp_tri(xy)
        
        # Interpolate horizontally
        tmp = Regridder.interpolate(original_o.ravel(),tri,xyTARG)
        tmp = tmp.reshape(targSY,targSX)
        # Mask invalid data, extrapolation and land points
        tmp = np.ma.masked_greater(tmp,np.nanmax(original_o))
        tmp = np.ma.masked_less(tmp,np.nanmin(original_o))
        #tmp = np.ma.masked_where(target_mask==0,tmp)
        tmp = np.ma.masked_invalid(tmp)

        original_on_target_grid = tmp
        
        return original_on_target_grid

    @staticmethod
    def interp_tri(xy):
        tri = qhull.Delaunay(xy)
        return tri

    @staticmethod
    def interpolate(values, tri,uv,d=2):
        simplex = tri.find_simplex(uv)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uv - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        return np.einsum('nj,nj->n', np.take(values, vertices),  np.hstack((bary, 1.0 - bary.sum(axis=1, keepdims=True))))
    
    @staticmethod
    def regrid_xr_dataarray(da,target_lon,target_lat):
        """
        author: Eike Koehn
        date: June 15, 2024
        description: Regrids an xarray dataarray from a curvilinear grid to another specified grid. Works only if the original horizontal coordinates are called 'lat' and 'lon'.
        """
        print(da)

        da_lon = da.lon.values
        da_lat = da.lat.values

        new_shape = list(da.shape)
        new_shape[-2] = np.size(target_lat)
        new_shape[-1] = np.size(target_lon)
        new_shape = tuple(new_shape)

        dims = list(da.dims)
        if 'eta_rho' in dims and 'xi_rho' in dims:
            dims.remove('eta_rho')
            dims.remove('xi_rho')
            dims.append('lat')
            dims.append('lon')

        # set up interpolated data array
        data_regridded = np.zeros(new_shape)+np.NaN
        rcoords = da.coords.to_dataset().to_dict()['coords']
        rcoords2 = dict()
        for key in rcoords.keys():
            rcoords2[key] = rcoords[key]['data']
        rcoords2['lat'] = target_lat # (["eta_rho","xi_rho"],target_lat)
        rcoords2['lon'] = target_lon # (["eta_rho","xi_rho"],target_lon)

        print(rcoords2)

        regridded_da = xr.DataArray(data_regridded, dims=dims, coords=rcoords2)

        print('Need to loop over some dimension because regridding routine can only handle 2d data.')
        coords_to_loop_over = list(da.coords)
        coords_to_loop_over.remove('lat')
        coords_to_loop_over.remove('lon')

        number_of_loops = len(coords_to_loop_over)

        if number_of_loops == 0:
            dslice = da.values
            regridded_da[:] = Regridder.regrid_original_to_target(dslice,da_lon,da_lat,target_lon,target_lat)#

        elif number_of_loops == 1: # need to loop only over 1 dimension
            coord = coords_to_loop_over[0]
            #print(coord)
            for ddx,dep in enumerate(da[coord]):
                print(ddx,int(dep))
                dslice = da.isel({coord:ddx}).values
                #print(dslice)
                regridded_dslice = Regridder.regrid_original_to_target(dslice,da_lon,da_lat,target_lon,target_lat)#
                regridded_da.isel({coord:ddx})[:] = regridded_dslice

        return regridded_da
# %%
