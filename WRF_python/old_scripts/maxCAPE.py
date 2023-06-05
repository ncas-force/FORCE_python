import numpy as np
import cartopy
from cartopy import crs
from cartopy.feature import NaturalEarthFeature
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import xarray as xr
import os
import sys
import ninept_smoother
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from wrf import (getvar, interplevel, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs)

# Define destination direction
dest_dir = "/home/earajr/FORCE_WRF_plotting/output/2dcapecin"
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

# Input WRF out file as an argument (full path)
wrf_fil = sys.argv[1]

# Check for existance of WRF out file
if not os.path.exists(wrf_fil):
    raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
wrf_in= Dataset(wrf_fil)
times = extract_times(wrf_in, ALL_TIMES)

# Extract the number of times within the WRF file and loop over all times in file
num_times = np.size(extract_times(wrf_in, ALL_TIMES))

for i in np.arange(0, num_times, 1):

# Read pressure geopotential and winds

   cape_2d = getvar(wrf_in, 'cape_2d', timeidx=i)
   rh = getvar(wrf_in, 'rh', timeidx=i)
   height = getvar(wrf_in, 'height_agl')

   max_cape = cape_2d[0]
   max_cin = np.where(cape_2d[1] >= 0, cape_2d[1]*-1.0, 0.0)
   lcl = cape_2d[2]
   lfc = cape_2d[3]

   interp_levs = np.linspace(lcl, lfc, num=10)
   rh_interp = []
   
   for j in np.arange(0,10):
      rh_interp.append(interplevel(rh, height, interp_levs[j]))

   rh_mean = np.nanmean(rh_interp, axis=0)
   
# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) 
   cart_proj = get_cartopy(cape_2d)
   lats, lons = latlon_coords(cape_2d)

# Create figure and axes
   fig = plt.figure(figsize=(10,10))
   ax = plt.axes(projection=cart_proj)
   ax.coastlines(linewidth=0.5)

# Plot mean RH between LCL and LFC

   rh_lvls = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
   RH = plt.contourf(lons, lats, rh_mean, levels=rh_lvls, colors='none', zorder=2,  hatches=['','','','','','','.','.','.','.'], transform=crs.PlateCarree())

# Plot CIN

   cin_lvls = [-250, -150, -100, -50]
   CIN = plt.contour(lons, lats, max_cin, levels=cin_lvls, colors='red', linestyles='solid', zorder=3, transform=crs.PlateCarree())
   plt.clabel(CIN, CIN.levels, inline=True, fmt= '%1.0f')


# Plot CAPE

   cape_lvls = [60, 80, 100, 120, 150, 175, 200, 240, 290, 340, 400, 500, 600, 700, 850, 1000, 1250, 1500, 1750, 2000, 2500]
   plt.contourf(lons, lats, max_cape, levels=cape_lvls, cmap='magma_r', zorder=1, transform=crs.PlateCarree())

   if np.size(lats[:,0]) < np.size(lats[0,:]):
      portrait = True
   else:
      portrait = False

   print(np.size(lats[:,0]))
   print(np.size(lats[0,:]))

#  portrait = False

# Create inset colourbar

   if portrait:
      cbbox = inset_axes(ax, '13%', '90%', loc = 7)
      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
      cbbox.set_facecolor([1,1,1,0.7])
      cbbox.text(0.7,0.5, "Maximum CAPE (J/kg)", rotation=90.0, verticalalignment='center', horizontalalignment='center')
      cbbox.text(0.85,0.25, "Maximum CIN (J/kg)", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='red')
      cbbox.text(0.85,0.75, u'$\u25CF$'+" RH between LCL and LFC (over 60%)", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='black')
      cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
      cb = plt.colorbar(cax=cbaxes, aspect=20)
   else:
      cbbox = inset_axes(ax, '90%', '12%', loc = 8)
      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
      cbbox.set_facecolor([1,1,1,0.7])
      cbbox.text(0.5,0.3, "Maximum CAPE (J/kg)", verticalalignment='center', horizontalalignment='center')
      cbbox.text(0.75,0.15, u'$\u25CF$'+" RH between LCL and LFC (over 60%)", verticalalignment='center', horizontalalignment='center', color='black')
      cbbox.text(0.25,0.15, "Maximum CIN (J/kg)", verticalalignment='center', horizontalalignment='center', color='red')
      cbaxes = inset_axes(cbbox, '95%', '30%', loc = 9)
      cb = plt.colorbar(cax=cbaxes, orientation='horizontal')

# Add inset timestamp
   tsbox = inset_axes(ax, '95%', '3%', loc = 9)
   [tsbox.spines[k].set_visible(False) for k in tsbox.spines]
   tsbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
   tsbox.set_facecolor([1,1,1,1])
   
   sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')
   valid_time = str(extract_times(wrf_in, ALL_TIMES)[i])[0:22]

#   print(sim_start_time['SIMULATION_START_DATE'])

   tsbox.text(0.01, 0.45, "Start date: "+sim_start_time['SIMULATION_START_DATE'], verticalalignment='center', horizontalalignment='left')
   tsbox.text(0.99, 0.45, "Valid_date: "+valid_time, verticalalignment='center', horizontalalignment='right')

## Add wind vectors after thinning.
#   thin = [int(x/15.) for x in lons.shape]
#   ax.quiver(to_np(lons[::thin[0],::thin[1]]), to_np(lats[::thin[0],::thin[1]]), to_np(u10[::thin[0],::thin[1]]), to_np(v10[::thin[0],::thin[1]]), pivot='middle', transform=crs.PlateCarree())
#
# Save image
   
   grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']
   
   plt.savefig(dest_dir+"/capelcllfcrh_d0"+str(grid_id)+"_"+sim_start_time['SIMULATION_START_DATE']+"_valid_"+valid_time[0:16]+".png", bbox_inches='tight')
   plt.close()
