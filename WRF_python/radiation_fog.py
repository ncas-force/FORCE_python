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
import calc_gradient

from wrf import (getvar, interplevel, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs)

# Calculation of the risk of radiation fog (and the consequent risk of freezing fog) based on the Craddock and Pritchard method.

# Define destination direction
dest_dir = "/home/earajr/FORCE_WRF_plotting/output/radiationfog"
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

# Input WRF out file as an argument (full path)
wrf_fil = sys.argv[1]
path = os.path.dirname(wrf_fil)
wrf_fil_head = os.path.basename(wrf_fil)[0:11]

YYYY = os.path.basename(wrf_fil)[11:15]
MM = os.path.basename(wrf_fil)[16:18]
DD = os.path.basename(wrf_fil)[19:21]
hh = os.path.basename(wrf_fil)[22:24]
mm = os.path.basename(wrf_fil)[25:27]
ss = os.path.basename(wrf_fil)[28:30]

for i in np.arange(0, 24, 1):
   if (datetime(int(YYYY), int(MM), int(DD), int(hh))-timedelta(hours=int(i))).strftime("%H") == "12" :
      wrf_fil2 =path+"/"+wrf_fil_head+(datetime(int(YYYY), int(MM), int(DD), int(hh), int(mm), int(ss))-timedelta(hours=int(i))).strftime("%Y-%m-%d_%H:%M:%S")

# Check for existance of WRF out file
if not os.path.exists(wrf_fil2):
    raise ValueError("Warning! "+wrf_fil2+" does not exist, cannot calculate fog risk.")

# Read WRF out netcdf
wrf_in2= Dataset(wrf_fil2)
times2 = extract_times(wrf_in2, ALL_TIMES)
num_times2 = np.size(extract_times(wrf_in2, ALL_TIMES))

for i in np.arange(0, num_times2, 1):
    if (times2[i].astype(str)[11:13]) == "12" and (times2[i].astype(str)[14:16]) == "00" and (times2[i].astype(str)[17:19]) == "00":
       noon_index = i

t2_12 = getvar(wrf_in2, 'T2', timeidx=noon_index)
t2_12 = t2_12 - 273.15

td2_12 = getvar(wrf_in2, 'td2', timeidx=noon_index, units='degC')

Y = (0.044*t2_12) + (0.844*td2_12) - 0.55

wrf_in= Dataset(wrf_fil)
times = extract_times(wrf_in, ALL_TIMES)
num_times = np.size(extract_times(wrf_in, ALL_TIMES))

for i in np.arange(0, num_times, 1):

   cloud_frac = getvar(wrf_in, 'cloudfrac', timeidx=i)
   cloud_frac_max = np.amax(cloud_frac, axis=0)

   cloud_frac_max_okt = cloud_frac_max * 8.0

   lats, lons = latlon_coords(cloud_frac)

   f = 2.0*0.000072921*np.sin((lats*2*np.pi)/360.0)

   g = np.full_like(f, 9.81)

   z = getvar(wrf_in, 'z', timeidx=i, units='m')
   pressure = getvar(wrf_in, 'pressure', timeidx=i)
   z_900 = interplevel(z, pressure, 900.0)

   dzdn_900 = calc_gradient.calc_gradient(z_900, to_np(lats), to_np(lons))

   dzdn_900_mag = np.sqrt((dzdn_900[0]**2.0) + (dzdn_900[1]**2.0))

   Vg = (dzdn_900_mag*g)/f

   A = np.zeros_like(Vg, dtype='float')
   A = np.where((Vg <= 12.5) & (cloud_frac_max_okt > 4), A+1.0, A)
   A = np.where((Vg <= 12.5) & (cloud_frac_max_okt > 6), A+0.5, A)

   A = np.where((Vg > 12.5) & (cloud_frac_max_okt <= 2), A-1.5, A)
   A = np.where((Vg > 12.5) & (cloud_frac_max_okt > 4), A+0.5, A)

   Tf = Y+A

   T2 = getvar(wrf_in, 'T2', timeidx=i)

   T2_C = T2-273.15

   Tdiff = Tf - T2_C

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) 
   cart_proj = get_cartopy(z)
   lats, lons = latlon_coords(z)

# Create figure and axes
   fig = plt.figure(figsize=(10,10))
   ax = plt.axes(projection=cart_proj)
   ax.coastlines(linewidth=0.5)
   ax.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
   ax.add_feature(cartopy.feature.LAND,facecolor=("sandybrown"))

# Plot cloud cover

   cloud_lvls = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
   color1 = [1,1,1,0]
   color2 = [1,1,1,1]
   cloud_cmap = np.linspace(color1, color2, 7)
   cloud = plt.contourf(lons, lats, Tdiff, levels=cloud_lvls, colors=cloud_cmap, zorder=1, antialiased=True, transform=crs.PlateCarree(), extend='max')

   if np.size(lats[:,0]) < np.size(lats[0,:]):
      portrait = True
   else:
      portrait = False

# Create inset colourbar

   if portrait:
      cbbox = inset_axes(ax, '10%', '90%', loc = 7)
      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
      cbbox.set_facecolor([1,1,1,0.7])
      cbbox.text(0.8,0.5, "Radiation fog (temperature below fog point)", rotation=90.0, verticalalignment='center', horizontalalignment='center')
#      cbbox.text(0.85,0.25, "Cloud cover (white background shading)", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='black')
#      cbbox.text(0.85,0.75, u'$\u00D7$'+" Chance of snow", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='black')
      cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
      cb = plt.colorbar(cax=cbaxes, aspect=20)
   else:
      cbbox = inset_axes(ax, '90%', '10%', loc = 8)
      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
      cbbox.set_facecolor([1,1,1,0.7])
      cbbox.text(0.5,0.2, "Radiation fog (temperature below fog point)", verticalalignment='center', horizontalalignment='center')
#      cbbox.text(0.75,0.15, u'$\u00D7$'+" Chance of snow", verticalalignment='center', horizontalalignment='center', color='black')
#      cbbox.text(0.25,0.15, "Cloud cover (white background shading)", verticalalignment='center', horizontalalignment='center', color='black')
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
   
   plt.savefig(dest_dir+"/radiationfog_d0"+str(grid_id)+"_"+sim_start_time['SIMULATION_START_DATE']+"_valid_"+valid_time[0:16]+".png", bbox_inches='tight')
   plt.close()

