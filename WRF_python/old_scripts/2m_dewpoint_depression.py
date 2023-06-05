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
dest_dir = "/home/earajr/FORCE_WRF_plotting/output/2mdewpointdepression"
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

   ht = getvar(wrf_in, 'z', timeidx=i)
   t2 = getvar(wrf_in, 'T2', timeidx=i)
   td2 = getvar(wrf_in, 'td2', timeidx=i)
   u10 = getvar(wrf_in, 'uvmet10', timeidx=i)[0]
   v10 = getvar(wrf_in, 'uvmet10', timeidx=i)[1]

   tdp2 = t2 - 273.15 - td2
   tdp2 = np.where(tdp2 > 5.0, 5.0, tdp2)
   
# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) 
   cart_proj = get_cartopy(t2)
   lats, lons = latlon_coords(t2)

# Create figure and axes
   fig = plt.figure(figsize=(10,10))
   ax = plt.axes(projection=cart_proj)
   ax.coastlines(linewidth=0.5)

# Plot geopotential height at 10 dam intervals
   tdp2_lvl = np.arange(0.0, 5.2, 0.2)
   plt.contourf(lons, lats, tdp2, levels=tdp2_lvl, cmap='magma_r', transform=crs.PlateCarree())

# Identify whether domain is portrait or landscape

   if np.size(lats[:,0]) < np.size(lats[0,:]):
      portrait = True
   else:
      portrait = False

#  portrait = False

# Create inset colourbar

   if portrait:
      cbbox = inset_axes(ax, '13%', '90%', loc = 7)
      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
      cbbox.set_facecolor([1,1,1,0.7])
      cbbox.text(0.7,0.5, "2m dewpoint depression (K)", rotation=90.0, verticalalignment='center', horizontalalignment='center')
      cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
      cb = plt.colorbar(cax=cbaxes, aspect=20)
   else:
      cbbox = inset_axes(ax, '90%', '12%', loc = 8)
      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
      cbbox.set_facecolor([1,1,1,0.7])
      cbbox.text(0.5,0.3, "2m dewpoint depression (K)", verticalalignment='center', horizontalalignment='center')
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

# Add wind vectors after thinning.
   thin = [int(x/15.) for x in lons.shape]
   ax.quiver(to_np(lons[::thin[0],::thin[1]]), to_np(lats[::thin[0],::thin[1]]), to_np(u10[::thin[0],::thin[1]]), to_np(v10[::thin[0],::thin[1]]), pivot='middle', transform=crs.PlateCarree())

# Save image

   grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']

   plt.savefig(dest_dir+"/2mdewpointdepression_d0"+str(grid_id)+"_"+sim_start_time['SIMULATION_START_DATE']+"_valid_"+valid_time[0:16]+".png", bbox_inches='tight')
   plt.close()
