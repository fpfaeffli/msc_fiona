#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys

#%%

class GetRegions():
    """
    author: Eike E. Koehn
    date: June 7, 2024
    description: This class contains a number of functions that enable the definition of the different study regions. 
    """ 

    @staticmethod
    def define_CalCS_regions(lon,lat,d2coast):

        if np.size(np.shape(lon)) == 1 and np.size(np.shape(lat)) == 1:
            coord_dims = '1d'
            lon_1d = lon.data*1.
            lat_1d = lat.data*1.
            lon,lat = np.meshgrid(lon,lat)
        elif np.size(np.shape(lon)) == 2 and np.size(np.shape(lat)) == 2:
            coord_dims = '2d'
        else:
            raise Exception('Not implemented yet.')
            #obs_area = xr.DataArray(area,coords={'lat':(["y","x"],lat),'lon':(["y","x"],lon)}, dims=["y","x"])

        boundary_latitudes = dict()
        boundary_latitudes['all_lats'] = dict()
        boundary_latitudes['all_lats']['minlat'] = 30
        boundary_latitudes['all_lats']['maxlat'] = 47
        boundary_latitudes['southern'] = dict()
        boundary_latitudes['southern']['minlat'] = 30
        boundary_latitudes['southern']['maxlat'] = 34.7
        boundary_latitudes['central'] = dict()
        boundary_latitudes['central']['minlat'] = 34.7
        boundary_latitudes['central']['maxlat'] = 40.5
        boundary_latitudes['northern'] = dict()
        boundary_latitudes['northern']['minlat'] = 40.5
        boundary_latitudes['northern']['maxlat'] = 47

        boundary_d2coasts = dict()
        boundary_d2coasts['all_dists'] = dict()
        boundary_d2coasts['all_dists']['mindist'] = 0
        boundary_d2coasts['all_dists']['maxdist'] = 370.4 #500
        boundary_d2coasts['coastal'] = dict()
        boundary_d2coasts['coastal']['mindist'] = 0
        boundary_d2coasts['coastal']['maxdist'] = 100
        # boundary_d2coasts['transition'] = dict()
        # boundary_d2coasts['transition']['mindist'] = 100
        # boundary_d2coasts['transition']['maxdist'] = 300
        boundary_d2coasts['offshore'] = dict()
        boundary_d2coasts['offshore']['mindist'] = 100   # 300
        boundary_d2coasts['offshore']['maxdist'] = 370.4 #500

        regions_dict = dict()
        for boundary_d2coast in boundary_d2coasts.keys():
            for boundary_latitude in boundary_latitudes.keys():
                regions_dict[f'{boundary_d2coast}_{boundary_latitude}'] = dict()
                regions_dict[f'{boundary_d2coast}_{boundary_latitude}']['mindist'] = boundary_d2coasts[boundary_d2coast]['mindist']
                regions_dict[f'{boundary_d2coast}_{boundary_latitude}']['maxdist'] = boundary_d2coasts[boundary_d2coast]['maxdist']
                regions_dict[f'{boundary_d2coast}_{boundary_latitude}']['minlat'] = boundary_latitudes[boundary_latitude]['minlat']
                regions_dict[f'{boundary_d2coast}_{boundary_latitude}']['maxlat'] = boundary_latitudes[boundary_latitude]['maxlat']

        # now actually create the respective dataset
        lon_cutoff = 225 
        for region in regions_dict.keys():
            region_mask = (lon>lon_cutoff) * (lon<245.05) * (lat>regions_dict[region]['minlat']) * (lat<=regions_dict[region]['maxlat']) * (d2coast>regions_dict[region]['mindist']) * (d2coast<=regions_dict[region]['maxdist'])
            region_mask = xr.where(region_mask==0.,np.NaN,region_mask)
            #region_mask[region_mask==0.]=np.NaN
            #print(region_mask)
            if coord_dims == '1d':
                regions_dict[region]['mask'] = xr.DataArray(region_mask.data,coords={'lat':(["lat"],lat_1d),'lon':(["lon"],lon_1d)}, dims=["lat","lon"])
            elif coord_dims == '2d':
                regions_dict[region]['mask'] = region_mask
            
        # now add a mask for the full Northeast Pacific that is usually mapped
        regions_dict['full_map'] = dict()
        regions_dict['full_map']['minlat'] = 27#20
        regions_dict['full_map']['maxlat'] = 55#60
        regions_dict['full_map']['minlon'] = 226#225
        regions_dict['full_map']['maxlon'] = 250#246#255
        region_mask = (lat>regions_dict['full_map']['minlat']) * (lat<=regions_dict['full_map']['maxlat']) * (lon>regions_dict['full_map']['minlon']) * (lon<=regions_dict['full_map']['maxlon'])
        region_mask = xr.where(np.isnan(d2coast),np.NaN,region_mask)
        region_mask = xr.where(region_mask==0.,np.NaN,region_mask)
        regions_dict['full_map']['mask'] = region_mask

        # now that we have all regions defined, I go ahead to give each region an id colour, i.e. the id_colour
        for region in regions_dict.keys():
            # set the colour based on the distance to the coast
            if 'offshore' in region:
                id_colour = '#1f77b4' # 'C0' blue
            # elif 'transition' in region:
            #     id_colour = '#ff7f0e' # 'C1' orange
            elif 'coastal' in region:
                id_colour = '#ff7f0e' # 'C1' orange '#2ca02c' # 'C2' green
            elif region == 'all_dists_all_lats':
                id_colour = '#d62728' # 'C3' red
            elif region == 'full_map':
                id_colour = '#9467bd' # 'C4' purple
            # vary the colours based on the latitude band
            if 'all_lats' in region:
                id_colour = id_colour # 'C0' blue
            elif 'northern' in region:
                id_colour = GetRegions.adjust_lightness(id_colour, amount=0.5)
            elif 'central' in region:
                id_colour = GetRegions.adjust_lightness(id_colour, amount=0.75)
            elif 'southern' in region:
                id_colour = GetRegions.adjust_lightness(id_colour, amount=1.25) 
            # put into the dictionary
            regions_dict[region]['id_colour'] = id_colour

        return regions_dict

    def adjust_lightness(color, amount=0.5):
        """
        gotten from: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
        """
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        c_rgb = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
        c_rgb = tuple([int(255*x) for x in c_rgb])
        return '#%02x%02x%02x' % c_rgb
    

# %%
# fig, ax = plt.subplots()
# for region in model_regions_dict.keys():
#     if region != 'full_map':# and 'all' not in region:
#         cmap = mc.ListedColormap([model_regions_dict[region]['id_colour']])
#         bounds = [0.5, 1.5]  # Define bounds such that 1 is within the range
#         norm = mc.BoundaryNorm(bounds, cmap.N)
#         plt.pcolormesh(model_regions_dict[region]['mask'],cmap=cmap,norm=norm)
# plt.show()
# %%
