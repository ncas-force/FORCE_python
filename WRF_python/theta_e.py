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

# Input WRF out file and pressure levels as arguments (full path)

wrf_fil = sys.argv[1]
plevs = sys.argv[2:]

# Define destination direction
for plev in plevs:
   dest_dir = "/home/earajr/FORCE_WRF_plotting/output/eth/"+str(plev)
   print(dest_dir)
   if not os.path.isdir(dest_dir):
      os.makedirs(dest_dir)

# Check for existance of WRF out file
if not os.path.exists(wrf_fil):
    raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
wrf_in= Dataset(wrf_fil)

# Extract the number of times within the WRF file and loop over all times in file
num_times = np.size(extract_times(wrf_in, ALL_TIMES))

for i in np.arange(0, num_times, 1):

# Read pressure geopotential and winds
   pres = getvar(wrf_in, 'pressure', timeidx=i)
   theta_e = getvar(wrf_in, 'theta_e', timeidx=i)

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) 
   cart_proj = get_cartopy(theta_e)
   lats, lons = latlon_coords(theta_e)

# Loop over pressure levels and check that they are sensible

   for j in np.arange(0, np.shape(plevs)[0], 1):
      dest_dir = "/home/earajr/FORCE_WRF_plotting/output/eth/"+str(plevs[j])

# Interpolate to current pressure level

      theta_e_plev = interplevel(theta_e, pres, plevs[j])

# Create figure and axes
      fig = plt.figure(figsize=(10,10))
      ax = plt.axes(projection=cart_proj)
      ax.coastlines(linewidth=0.5)

# Plot theta_e
      theta_e_lvl = np.arange(280.0, 350.0, 1.0)
      plt.contourf(lons, lats, theta_e_plev, levels=theta_e_lvl, cmap='CMRmap', transform=crs.PlateCarree())

# Identify whether domain is portrait or landscape

      if np.size(lats[:,0]) < np.size(lats[0,:]):
         portrait = True
      else:
         portrait = False

# Create inset colourbar

      if portrait:
         cbbox = inset_axes(ax, '13%', '90%', loc = 7)
         [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
         cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
         cbbox.set_facecolor([1,1,1,0.7])
#         cbbox.text(0.7,0.5, "200 hPa windspeed (m/s)", rotation=90.0, verticalalignment='center', horizontalalignment='center')
#         cbbox.text(0.85,0.5, "Geopotential height (10 dm spacing)", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='red')
         cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
         cb = plt.colorbar(cax=cbaxes, aspect=20)
      else:
         cbbox = inset_axes(ax, '90%', '12%', loc = 8)
         [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
         cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
#         cbbox.set_facecolor([1,1,1,0.7])
#         cbbox.text(0.5,0.3, "200 hPa windspeed (m/s)", verticalalignment='center', horizontalalignment='center')
         cbbox.text(0.5,0.15, "Geopotential height (10 dm spacing)", verticalalignment='center', horizontalalignment='center', color='red')
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

# Save image

      grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']
      
      plt.savefig(dest_dir+"/eth"+str(plevs[j])+"_d0"+str(grid_id)+"_"+sim_start_time['SIMULATION_START_DATE']+"_valid_"+valid_time[0:16]+".png", bbox_inches='tight')
      plt.close()
