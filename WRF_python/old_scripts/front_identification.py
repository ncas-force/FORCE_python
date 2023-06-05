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
import calc_gradient
import calc_divergence
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pint import UnitRegistry
from skimage.morphology import skeletonize, thin
from skimage.filters import sato, gaussian
from matplotlib import colors

from wrf import (getvar, interplevel, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs)

# Define destination direction
dest_dir = "/home/earajr/FORCE_WRF_plotting/output/T2fronts"
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

# Input WRF out file as an argument (full path)
wrf_fil = sys.argv[1]

# Check for existance of WRF out file
if not os.path.exists(wrf_fil):
    raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
wrf_in= Dataset(wrf_fil)

# Read grid spacing and scale gaussian sigma value based on grid spacing
dx = extract_global_attrs(wrf_in, 'DX')['DX']

if dx < 20000:
   sigma = 5.875 - 0.21875*(dx/1000.0)
else:
   sigma = 1.5

# Extract the number of times within the WRF file and loop over all times in file
num_times = np.size(extract_times(wrf_in, ALL_TIMES))

for i in np.arange(0, num_times, 1):

# Read pressure, 2m temperature, winds, lats and lons and projection

   pres = getvar(wrf_in, 'pressure', timeidx=i)
   T2 = getvar(wrf_in, 'T2', timeidx=i)
   theta_e = getvar(wrf_in, 'theta_e', timeidx=i)
   v = getvar(wrf_in, 'va', timeidx=i, units='m/s')
   u = getvar(wrf_in, 'ua', timeidx=i, units='m/s')
   lats, lons = latlon_coords(T2)
   cart_proj = get_cartopy(T2)
 
# Calculate theta_w and interpolate theta_w and winds to required pressure level and calculate gradient

   plev = 800.0

   temp_theta_e = interplevel(theta_e, pres, plev)
   temp_theta_w = 45.114 - (51.489*(273.15/temp_theta_e)**3.504)  # calculate theta_w from theta e using Scheid method
   temp_grad = calc_gradient.calc_gradient(temp_theta_w, to_np(lats), to_np(lons))
   temp_grad_mag = np.sqrt((temp_grad[0]**2.0) + (temp_grad[1]**2.0))
   theta_w_plev_grad_temp = temp_grad_mag.to("1 / kilometer")
   theta_w_plev_grad = gaussian(theta_w_plev_grad_temp, sigma=sigma)

   v_plev = interplevel(v, pres, plev)
   u_plev = interplevel(u, pres, plev)

# Start the front identification process

# Loop through a wide range of thresholds for theta w gradient (this should allow for weaker gradients from lower resolution simulations to be captured
# for each thresholded image the resultant binary is thinned to contiguous regions, all thresholded and thinned binaries are summed.

   for thresh in np.arange(0.001, 0.08, 0.001):

      front_id_bin = np.where((theta_w_plev_grad >= thresh), 1, 0)
      front_thinned = np.where(thin(front_id_bin) == True, 1, 0)
      if thresh == 0.001:
         sum_thinned = front_thinned
      else:
         sum_thinned = sum_thinned + front_thinned

# Apply sato ridge finding filter to the sum of the thinned thresholded regions, threshold the sato result at a very low level (so not to lose significant structure and reapply thinning to resltant binary

   front_loc = sato(sum_thinned, black_ridges=0)
   front_loc_bin = np.where(front_loc > 2e-19, 1, 0)
   front_loc_thinned = np.where(thin(front_loc_bin) == True, 1, 0) 

# Convert thinned front location binary to poly lines to be drawn on once other variables are plotted

   import trace_skeleton

   polys = trace_skeleton.from_numpy(front_loc_thinned)
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

# Create figure and axes
   fig = plt.figure(figsize=(10,10))
   ax = plt.axes(projection=cart_proj)
   ax.coastlines(linewidth=0.5)

# Plot 2 m temperature 
   T2 = T2 -273.15
   T2_lvl = np.arange(-15.0, 30.0, 1.0)

   plt.contourf(lons, lats, T2, levels=T2_lvl, cmap='viridis', zorder=1, transform=crs.PlateCarree())

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
      cbbox.text(0.7,0.5, "2m temperature (C) and 2m wind vectors", rotation=90.0, verticalalignment='center', horizontalalignment='center')
      cbbox.text(0.85,0.5, "Likely front position (based on 800 hPa wetbulb potential temperature)", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='red')
      cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
      cb = plt.colorbar(cax=cbaxes, aspect=20)
   else:
      cbbox = inset_axes(ax, '90%', '12%', loc = 8)
      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
      cbbox.set_facecolor([1,1,1,0.7])
      cbbox.text(0.5,0.3, "2m temperature (C) and 2m wind vectors", verticalalignment='center', horizontalalignment='center')
      cbbox.text(0.5,0.15, "Likely front position (based on 800 hPa wetbulb poteential temperature)", verticalalignment='center', horizontalalignment='center', color='red')
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
   thin_vec = [int(x/15.) for x in lons.shape]
   ax.quiver(to_np(lons[::thin_vec[0],::thin_vec[1]]), to_np(lats[::thin_vec[0],::thin_vec[1]]), to_np(u_plev[::thin_vec[0],::thin_vec[1]]), to_np(v_plev[::thin_vec[0],::thin_vec[1]]), pivot='middle', transform=crs.PlateCarree())

# Add fronts using path
   
   for j in np.arange(0,np.shape(polys_lat)[0], 1):
      ax.plot(polys_lon[j], polys_lat[j], linewidth=2, color='red',transform=crs.PlateCarree())

# Save image 

   grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']

   plt.savefig(dest_dir+"/T2fronts_d0"+str(grid_id)+"_"+sim_start_time['SIMULATION_START_DATE']+"_valid_"+valid_time[0:16]+".png", bbox_inches='tight')
   plt.close()
