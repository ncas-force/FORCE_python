import numpy as np
import cartopy
from cartopy import crs
from cartopy.feature import NaturalEarthFeature
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import xarray as xr
import os
import sys
import ninept_smoother
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime, timedelta

from wrf import (getvar, interplevel, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs)

# Define destination direction
dest_dir = "/home/earajr/FORCE_WRF_plotting/output/clouddbz"
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

# Input WRF out file as an argument (full path)
wrf_fil = sys.argv[1]
#path = os.path.dirname(wrf_fil)
#wrf_fil_head = os.path.basename(wrf_fil)[0:11]
#
#YYYY = os.path.basename(wrf_fil)[11:15]
#MM = os.path.basename(wrf_fil)[16:18]
#DD = os.path.basename(wrf_fil)[19:21]
#hh = os.path.basename(wrf_fil)[22:24]
#mm = os.path.basename(wrf_fil)[25:27]
#ss = os.path.basename(wrf_fil)[28:30]
#
#wrf_fil2 =path+"/"+wrf_fil_head+(datetime(int(YYYY), int(MM), int(DD), int(hh), int(mm), int(ss))-timedelta(hours=1)).strftime("%Y-%m-%d_%H:%M:%S")

# Check for existance of WRF out file
if not os.path.exists(wrf_fil):
    raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
wrf_in= Dataset(wrf_fil)
times = extract_times(wrf_in, ALL_TIMES)

# Extract the number of times within the WRF file and loop over all times in file
num_times = np.size(extract_times(wrf_in, ALL_TIMES))

for i in np.arange(0, num_times, 1):

   cloud_frac = getvar(wrf_in, 'cloudfrac', timeidx=i)
   cloud_frac_max = np.amax(cloud_frac, axis=0)

   tc = getvar(wrf_in, 'tc', timeidx=i)
   pressure = getvar(wrf_in, 'pressure', timeidx=i)

   tc_850 = interplevel(tc, pressure, 850.0)

   mdbz = getvar(wrf_in, 'mdbz', timeidx=i)

#   if i == 0:
#
#      if os.path.exists(wrf_fil2):
#      
#         wrf_in2 = Dataset(wrf_fil2)
#         times2 = extract_times(wrf_in2, ALL_TIMES)
#         num_times2 = np.size(extract_times(wrf_in2, ALL_TIMES))
#      
#         rainc1 = getvar(wrf_in, 'RAINC', timeidx=i)
#         rainnc1 = getvar(wrf_in, 'RAINNC', timeidx=i)
#         snownc1 = getvar(wrf_in, 'SNOWNC', timeidx=i)
#         hailnc1 = getvar(wrf_in, 'HAILNC', timeidx=i)
#         graupel1 = getvar(wrf_in, 'GRAUPELNC', timeidx=i)
#
#         rainc2 = getvar(wrf_in2, 'RAINC', timeidx=num_times2-1)
#         rainnc2 = getvar(wrf_in2, 'RAINNC', timeidx=num_times2-1)
#         snownc2 = getvar(wrf_in2, 'SNOWNC', timeidx=num_times2-1)
#         hailnc2 = getvar(wrf_in2, 'HAILNC', timeidx=num_times2-1)
#         graupel2 = getvar(wrf_in2, 'GRAUPELNC', timeidx=num_times2-1)
#
#      else:
#         rainc1 = getvar(wrf_in, 'RAINC', timeidx=i)
#         rainnc1 = getvar(wrf_in, 'RAINNC', timeidx=i)
#         snownc1 = getvar(wrf_in, 'SNOWNC', timeidx=i)
#         hailnc1 = getvar(wrf_in, 'HAILNC', timeidx=i)
#         graupel1 = getvar(wrf_in, 'GRAUPELNC', timeidx=i)
#
#         rainc2 = rainc1 * 0.0
#         rainnc2 = rainnc1 * 0.0
#         snownc2 = snownc1 * 0.0
#         hailnc2 = hailnc1 * 0.0
#         graupel2 = graupel1 * 0.0
#
#
#   else:
#      rainc1 = getvar(wrf_in, 'RAINC', timeidx=i)
#      rainnc1 = getvar(wrf_in, 'RAINNC', timeidx=i)
#      snownc1 = getvar(wrf_in, 'SNOWNC', timeidx=i)
#      hailnc1 = getvar(wrf_in, 'HAILNC', timeidx=i)
#      graupel1 = getvar(wrf_in, 'GRAUPELNC', timeidx=i)
#
#
#      rainc2 = getvar(wrf_in, 'RAINC', timeidx=i-1)
#      rainnc2 = getvar(wrf_in, 'RAINNC', timeidx=i-1)
#      snownc2 = getvar(wrf_in, 'SNOWNC', timeidx=i-1)
#      hailnc2 = getvar(wrf_in, 'HAILNC', timeidx=i-1)
#      graupel2 = getvar(wrf_in, 'GRAUPELNC', timeidx=i-1)
#
#
#   rtot1hr = rainc1 + rainnc1 + snownc1 + hailnc1 + graupel1 - rainc2 - rainnc2 - snownc2 - hailnc2 - graupel2
#
#   tc_850 = np.where(rtot1hr <= 0.25, 0.0, tc_850)

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) 
   cart_proj = get_cartopy(mdbz)
   lats, lons = latlon_coords(mdbz)

# Create figure and axes
   fig = plt.figure(figsize=(10,10))
   ax = plt.axes(projection=cart_proj)
   ax.coastlines(linewidth=0.5)
   ax.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
   ax.add_feature(cartopy.feature.LAND,facecolor=("sandybrown"))

# Plot cloud cover

   cloud_lvls = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
   color1 = [1,1,1,0]
   color2 = [1,1,1,1]
   cloud_cmap = np.linspace(color1, color2, 16)
   cloud = plt.contourf(lons, lats, cloud_frac_max, levels=cloud_lvls, colors=cloud_cmap, zorder=1, antialiased=True, transform=crs.PlateCarree(), extend='max')

# Plot snow hatching

   snow_lvls = [-10.0, -5.0, 0.0]
   snow = plt.contourf(lons, lats, tc_850, levels=snow_lvls, colors='None', hatches=['XX','',''], zorder=3, transform=crs.PlateCarree())

# Plot precip

   precip_lvls = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
   cmap = mpl.cm.get_cmap('jet')
   cmap_sub = cmap(np.linspace(0.25,0.9, 10))
   plt.contourf(lons, lats, mdbz, levels=precip_lvls, colors=cmap_sub,  zorder=2, transform=crs.PlateCarree())

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
      cbbox.text(0.7,0.5, "Maximum reflectivity (dbz)", rotation=90.0, verticalalignment='center', horizontalalignment='center')
      cbbox.text(0.85,0.25, "Cloud cover (white background shading)", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='black')
      cbbox.text(0.85,0.75, u'$\u00D7$'+" Chance of snow", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='black')
      cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
      cb = plt.colorbar(cax=cbaxes, aspect=20)
   else:
      cbbox = inset_axes(ax, '90%', '12%', loc = 8)
      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
      cbbox.set_facecolor([1,1,1,0.7])
      cbbox.text(0.5,0.3, "Maximum reflectivity (dbz)", verticalalignment='center', horizontalalignment='center')
      cbbox.text(0.75,0.15, u'$\u00D7$'+" Chance of snow", verticalalignment='center', horizontalalignment='center', color='black')
      cbbox.text(0.25,0.15, "Cloud cover (white background shading)", verticalalignment='center', horizontalalignment='center', color='black')
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

   grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']
   
   plt.savefig(dest_dir+"/clouddbz_d0"+str(grid_id)+"_"+sim_start_time['SIMULATION_START_DATE']+"_valid_"+valid_time[0:16]+".png", bbox_inches='tight')
   plt.close()

