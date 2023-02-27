import numpy as np
import cartopy
from cartopy import crs
from cartopy.feature import NaturalEarthFeature
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import xarray as xr
import os
import ninept_smoother
import calc_gradient
import calc_divergence
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pint import UnitRegistry
from skimage.morphology import skeletonize, thin
from matplotlib import colors

from wrf import (getvar, interplevel, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs)

# Input WRF out file (this will be given as an argument when running operationally
#wrf_fil ="/home/earajr/example_WRF_output/wrfout_d01_2022-05-14_00:00:00"
wrf_fil ="../../example_WRF_output/wrfout_d01_2022-10-31_18:00:00"

# Check for existance of WRF out file
if not os.path.exists(wrf_fil):
    raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
wrf_in= Dataset(wrf_fil)

# Extract the number of times within the WRF file and loop over all times in file
num_times = np.size(extract_times(wrf_in, ALL_TIMES))

#for i in np.arange(0, num_times, 1):
for i in np.arange(6, 7, 1):

# Read 2m temperature, lats and lons

   pres = getvar(wrf_in, 'pressure', timeidx=i)
   T2 = getvar(wrf_in, 'T2', timeidx=i)
   tc = getvar(wrf_in, 'tc', timeidx=i)
   theta_e = getvar(wrf_in, 'theta_e', timeidx=i)
   v = getvar(wrf_in, 'va', timeidx=i, units='m/s')
   u = getvar(wrf_in, 'ua', timeidx=i, units='m/s')
   lats, lons = latlon_coords(tc)
 
# Interpolate temperature to 850 hPa

   tc_850 = interplevel(tc, pres, 850.0)
   theta_e_850 = interplevel(theta_e, pres, 850.0)
   v_850 = interplevel(v, pres, 850.0)
   u_850 = interplevel(u, pres, 850.0)

# Calculate theta_w_850 using the Scheid method

   theta_w_850 = 45.114 - (51.489*(273.15/theta_e_850)**3.504)

# Calculate T2 gradient

   T2_gradient = calc_gradient.calc_gradient(T2, to_np(lats), to_np(lons))
   T2_gradient_mag_kpm = np.sqrt((T2_gradient[0]**2.0) + (T2_gradient[1]**2.0))
   T2_gradient_mag = T2_gradient_mag_kpm.to("kelvin / kilometer")

# Calculate tc_850 gradient

   tc_850_gradient = calc_gradient.calc_gradient(tc_850, to_np(lats), to_np(lons))
   tc_850_gradient_mag_kpm = np.sqrt((tc_850_gradient[0]**2.0) + (tc_850_gradient[1]**2.0))
   tc_850_gradient_mag = tc_850_gradient_mag_kpm.to("kelvin / kilometer")

# Calculate theta_w_gradient

   theta_w_850_gradient = calc_gradient.calc_gradient(theta_w_850, to_np(lats), to_np(lons))
   theta_w_850_gradient_div = calc_divergence.calc_divergence(theta_w_850_gradient[0], theta_w_850_gradient[1], to_np(lats), to_np(lons))
   theta_w_850_gradient_mag =1000.0*(np.sqrt((theta_w_850_gradient[0]**2.0) + (theta_w_850_gradient[1]**2.0)))

   front_step1 = calc_gradient.calc_gradient(tc_850, to_np(lats), to_np(lons))
   front_step2 = (np.sqrt((front_step1[0]**2.0) + (front_step1[1]**2.0)))
   front_step3 = front_step2.to("kelvin / kilometer")
 #  front_step3 = ninept_smoother.smth9(front_step3, 0.5, 0.25)
 #  front_step3 = ninept_smoother.smth9(front_step3, 0.5, 0.25)
 #  front_step3 = ninept_smoother.smth9(front_step3, 0.5, 0.25)
   front_step3 = ninept_smoother.smth9(front_step3, 0.5, 0.25)
   front_step4 = np.where(front_step3 >= 0.025, front_step3, 0.0)
   front_step4 = ninept_smoother.smth9(front_step4, 0.5, 0.25)
   front_step4 = ninept_smoother.smth9(front_step4, 0.5, 0.25)
   front_step4 = ninept_smoother.smth9(front_step4, 0.5, 0.25)
   front_step4 = ninept_smoother.smth9(front_step4, 0.5, 0.25)
   front_step5 = np.where(front_step4 >= 0.01, 1.0, 0.0)
   front_step6 = np.where(thin(front_step5) == True, 1, 0)

   import trace_skeleton

   polys = trace_skeleton.from_numpy(front_step6)
   polys_lat = []
   polys_lon = []
   for poly in polys:
      lat_temp = []
      lon_temp = []
      for indices in poly:
          lat_temp.append(to_np(lats)[indices[1],indices[0]])
          lon_temp.append(to_np(lons)[indices[1],indices[0]])
      polys_lat.append(lat_temp)
      polys_lon.append(lon_temp)

#  front_step3 = calc_gradient.calc_gradient(front_step2, to_np(lats), to_np(lons))
#  front_step4 = calc_divergence.calc_divergence(front_step3[0], front_step3[1], to_np(lats), to_np(lons))

#   theta_w_850_gradient_mag = theta_w_850_gradient_mag_kpm.to("kelvin / kilometer")

## Apply smoothing multiple times to create more user friendly image
#   geop_height_200 = ninept_smoother.smth9(geop_height_200, 0.5, 0.25)
#   geop_height_200 = ninept_smoother.smth9(geop_height_200, 0.5, 0.25)
#   geop_height_200 = ninept_smoother.smth9(geop_height_200, 0.5, 0.25)
#   geop_height_200 = ninept_smoother.smth9(geop_height_200, 0.5, 0.25)

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) 
   cart_proj = get_cartopy(T2)

# Create figure and axes
   fig = plt.figure(figsize=(10,10))
   ax = plt.axes(projection=cart_proj)
   ax.coastlines(linewidth=0.5)

# Plot geopotential height at 10 dam intervals
   T2 = T2 -273.15
   T2_lvl = np.arange(-15.0, 30.0, 1.0)
   front_lvl = np.arange(0.75, 1.75, 0.5)
#   theta_w_850_gradient_mag_levs = np.arange(0.0, 0.05, 0.001)
#   theta_w_850_gradient_div_levs = np.arange(-0.5, 0.5, 0.02)
#   theta_w_850_levs = np.arange(0.0, 15.0, 0.25)

#   plt.contourf(lons, lats, front_step6, levels=front_lvl, linewidths=2, cmap=colors.ListedColormap([(0,0,0,0),(1,0,0,1)]), zorder=2, transform=crs.PlateCarree())
   plt.contourf(lons, lats, T2, levels=T2_lvl, cmap='viridis', zorder=1, transform=crs.PlateCarree())

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
      cbbox.text(0.7,0.5, "2m temperature (C) and 2m wind vectors", rotation=90.0, verticalalignment='center', horizontalalignment='center')
      cbbox.text(0.85,0.5, "Likely front position (base on 850 hPa temperature)", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='red')
      cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
      cb = plt.colorbar(cax=cbaxes, aspect=20)
   else:
      cbbox = inset_axes(ax, '90%', '12%', loc = 8)
      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
      cbbox.set_facecolor([1,1,1,0.7])
      cbbox.text(0.5,0.3, "2m temperature (C) and 2m wind vectors", verticalalignment='center', horizontalalignment='center')
      cbbox.text(0.5,0.15, "Likely front position (base on 850 hPa temperature)", verticalalignment='center', horizontalalignment='center', color='red')
      cbaxes = inset_axes(cbbox, '95%', '30%', loc = 9)
      cb = plt.colorbar(cax=cbaxes, orientation='horizontal')

# Add inset timestamp
   tsbox = inset_axes(ax, '95%', '3%', loc = 9)
   [tsbox.spines[k].set_visible(False) for k in tsbox.spines]
   tsbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
   tsbox.set_facecolor([1,1,1,1])

   sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')
   valid_time = str(extract_times(wrf_in, ALL_TIMES)[i])[0:22]

   tsbox.text(0.01, 0.45, "Start date: "+sim_start_time['SIMULATION_START_DATE'], verticalalignment='center', horizontalalignment='left')
   tsbox.text(0.99, 0.45, "Valid_date: "+valid_time, verticalalignment='center', horizontalalignment='right')

# Add wind vectors after thinning.
   thin = [int(x/15.) for x in lons.shape]
   ax.quiver(to_np(lons[::thin[0],::thin[1]]), to_np(lats[::thin[0],::thin[1]]), to_np(u_850[::thin[0],::thin[1]]), to_np(v_850[::thin[0],::thin[1]]), pivot='middle', transform=crs.PlateCarree())

# Add fronts using path

   print(polys_lat[0])
   print(polys_lon[0])
   
   for j in np.arange(0,np.shape(polys_lat)[0], 1):
      if np.size(polys_lon[j]) >= 5 :
         ax.plot(polys_lon[j], polys_lat[j], linewidth=2, color='red',transform=crs.PlateCarree())


# Save image 
   plt.savefig('fronts_method1.png', bbox_inches='tight')

