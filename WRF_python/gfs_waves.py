import numpy as np
import cartopy
from cartopy import crs
from cartopy.feature import NaturalEarthFeature
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import xarray as xr
import xesmf as xe
import os
import sys
import datetime
import ninept_smoother
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyArrowPatch, ArrowStyle

from wrf import (getvar, interplevel, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs)

# Define destination direction
dest_dir = "/home/earajr/FORCE_WRF_plotting/output/gfswaves"
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

# Input WRF out file as an argument (full path)
gfs_fil = sys.argv[1]
gfs_fil_base = os.path.basename(gfs_fil)
gfs_fil_dir = os.path.dirname(gfs_fil)

YYYY = gfs_fil_dir.split("/")[-1][0:4]
MM = gfs_fil_dir.split("/")[-1][4:6]
DD = gfs_fil_dir.split("/")[-1][6:8]
hh = gfs_fil_dir.split("/")[-1][8:10]

fore = gfs_fil_base.split(".")[4].split("_")[0].split("f")[1]

init_date = datetime.datetime(int(YYYY), int(MM), int(DD), int(hh), 0, 0)
fore_date = init_date + datetime.timedelta(hours=int(fore))

# Check for existance of WRF out file
if not os.path.exists(gfs_fil):
    raise ValueError("Warning! "+gfs_fil+" does not exist.")

# Read GFS netcdf

gfs_in = xr.open_dataset(gfs_fil)

#gfs_in= Dataset(gfs_fil)

#lats = gfs_in.variables['lat_0'][:]
#lons = gfs_in.variables['lon_0'][:]

swellwind_wave_height = gfs_in['HTSGW_P0_L1_GLL0']
primary_wave_direction = gfs_in['DIRPW_P0_L1_GLL0']
lats = gfs_in['lat_0']
lons = gfs_in['lon_0']
swellwind_wave_height = swellwind_wave_height.rename({'lat_0': 'latitude', 'lon_0': 'longitude'})

#wind_wave_height = gfs_in.variables['WVHGT_P0_L1_GLL0'][:,:]
#wind_wave_direction = gfs_in.variables['WVDIR_P0_L1_GLL0'][:,:]


#for dom in ["uk_d01", "uk_d02", "iceland_d01", "iceland_d02", "capeverde_d01", "capeverde_d02"]:
for dom in ["uk_d01"]:
   wrf_fil = "/home/earajr/FORCE_WRF_plotting/WRF_python/example_nwr_data/"+dom+".nc"

   wrf_in = Dataset(wrf_fil)

   t2 = getvar(wrf_in, 'T2', timeidx=0)
   dest_lats, dest_lons = latlon_coords(t2)
   cart_proj = get_cartopy(t2)

   t2 = t2.rename({'XLAT': 'latitude', 'XLONG' : 'longitude'})

   regridder_path = "/home/earajr/FORCE_WRF_plotting/WRF_python/example_nwr_data/gfs_to_wrf_"+dom+".nc"

   if os.path.exists(regridder_path):

      regridder = xe.Regridder(swellwind_wave_height, t2, 'nearest_s2d', weights=regridder_path)

   else:

      regridder = xe.Regridder(swellwind_wave_height, t2, 'nearest_s2d')
      regridder.to_netcdf(regridder_path)


   swellwind_wave_height_regridded = regridder(swellwind_wave_height)
   primary_wave_direction_regridded = regridder(primary_wave_direction)

   u = np.cos((270.0 - primary_wave_direction_regridded) * np.pi/180)
   v = np.sin((270.0 - primary_wave_direction_regridded) * np.pi/180)

   dest_lats = swellwind_wave_height_regridded['latitude']
   dest_lons = swellwind_wave_height_regridded['longitude']


   
   

# nwr projections
# uk +proj=merc +a=6370000.0 +b=6370000.0 +nadgrids=@null +lon_0=-2.0 +lat_ts=57.0 +k=1 +units=m +no_defs +type=crs
# iceland +proj=merc +a=6370000.0 +b=6370000.0 +nadgrids=@null +lon_0=-18.299999 +lat_ts=64.300003 +k=1 +units=m +no_defs +type=crs
# capeverde = +proj=merc +a=6370000.0 +b=6370000.0 +nadgrids=@null +lon_0=-24.1 +lat_ts=15.9 +k=1 +units=m +no_defs +type=crs 

#cart_proj = "+proj=merc +a=6370000.0 +b=6370000.0 +nadgrids=@null +lon_0=-2.0 +lat_ts=57.0 +k=1 +units=m +no_defs +type=crs"

# Create figure and axes
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=cart_proj)
ax.coastlines(linewidth=0.5)

# Plot 200 hPa windspeed    
wavehgt_lvl = np.arange(0, 10, 0.2)
plt.contourf(dest_lons, dest_lats, swellwind_wave_height_regridded, levels=wavehgt_lvl, cmap='jet', transform=crs.PlateCarree())

## Add wind vectors after thinning.
thin = [int(x/25.) for x in dest_lons.shape]
#ax.quiver(to_np(dest_lons[::thin[0],::thin[1]]), to_np(dest_lats[::thin[0],::thin[1]]), to_np(u[::thin[0],::thin[1]]), to_np(v[::thin[0],::thin[1]]), minlength=0.0, minshaft=0.0, scale=300,  pivot='tail', headwidth=20, headlength=2, headaxislength=1.5, transform=crs.PlateCarree())
ax.quiver(to_np(dest_lons[::thin[0],::thin[1]]), to_np(dest_lats[::thin[0],::thin[1]]), to_np(u[::thin[0],::thin[1]]), to_np(v[::thin[0],::thin[1]]), minlength=0.0, minshaft=0.0, scale=400,  pivot='tip', headwidth=20, headlength=2, headaxislength=1.5, transform=crs.PlateCarree(), color=[0,0,0,0.5])



'''
## Identify whether domain is portrait or landscape
#
#   if np.size(lats[:,0]) < np.size(lats[0,:]):
#      portrait = True
#   else:
#      portrait = False
#
##  portrait = False
#
## Create inset colourbar
#
#   if portrait:
#      cbbox = inset_axes(ax, '13%', '90%', loc = 7)
#      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
#      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
#      cbbox.set_facecolor([1,1,1,0.7])
#      cbbox.text(0.7,0.5, "200 hPa windspeed (m/s)", rotation=90.0, verticalalignment='center', horizontalalignment='center')
#      cbbox.text(0.85,0.5, "Geopotential height (10 dm spacing)", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='red')
#      cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
#      cb = plt.colorbar(cax=cbaxes, aspect=20)
#   else:
#      cbbox = inset_axes(ax, '90%', '12%', loc = 8)
#      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
#      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
#      cbbox.set_facecolor([1,1,1,0.7])
#      cbbox.text(0.5,0.3, "200 hPa windspeed (m/s)", verticalalignment='center', horizontalalignment='center')
#      cbbox.text(0.5,0.15, "Geopotential height (10 dm spacing)", verticalalignment='center', horizontalalignment='center', color='red')
#      cbaxes = inset_axes(cbbox, '95%', '30%', loc = 9)
#      cb = plt.colorbar(cax=cbaxes, orientation='horizontal')
#
## Add inset timestamp
#   tsbox = inset_axes(ax, '95%', '3%', loc = 9)
#   [tsbox.spines[k].set_visible(False) for k in tsbox.spines]
#   tsbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
#   tsbox.set_facecolor([1,1,1,1])
#
#   sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')
#   valid_time = str(extract_times(wrf_in, ALL_TIMES)[i])[0:22]
#
##   print(sim_start_time['SIMULATION_START_DATE'])
#
#   tsbox.text(0.01, 0.45, "Start date: "+sim_start_time['SIMULATION_START_DATE'], verticalalignment='center', horizontalalignment='left')
#   tsbox.text(0.99, 0.45, "Valid_date: "+valid_time, verticalalignment='center', horizontalalignment='right')
#
## Add wind vectors after thinning.
#   thin = [int(x/15.) for x in lons.shape]
#   ax.quiver(to_np(lons[::thin[0],::thin[1]]), to_np(lats[::thin[0],::thin[1]]), to_np(u_200[::thin[0],::thin[1]]), to_np(v_200[::thin[0],::thin[1]]), pivot='middle', transform=crs.PlateCarree())
#
## Save image
#
#   grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']
'''
plt.savefig(dest_dir+"/windwaves_d0unknown_"+init_date.strftime("%Y-%m-%d_%H:00:00")+"_valid_"+fore_date.strftime("%Y-%m-%dT%H:00:00")+".png", bbox_inches='tight')
plt.close()

