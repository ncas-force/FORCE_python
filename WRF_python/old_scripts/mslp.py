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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.measure import label
from pyproj import Geod

from wrf import (getvar, interplevel, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs)

# Define destination direction
dest_dir = "/home/earajr/FORCE_WRF_plotting/output/mslp"
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

# Define geodesy
geod = Geod(ellps='WGS84')

# Extract the number of times within the WRF file and loop over all times in file
num_times = np.size(extract_times(wrf_in, ALL_TIMES))

for i in np.arange(0, num_times, 1):
#for i in np.arange(0, 3, 1):

# Read pressure geopotential and winds
   slp = getvar(wrf_in, 'slp', timeidx=i)
   ter = getvar(wrf_in, 'ter', timeidx=i)
   slp_smooth = ninept_smoother.smth9(slp, 0.5, 0.25)
   slp_smooth = ninept_smoother.smth9(slp_smooth, 0.5, 0.25)
   slp_smooth = ninept_smoother.smth9(slp_smooth, 0.5, 0.25)

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) 
   cart_proj = get_cartopy(slp)
   lats, lons = latlon_coords(slp)

# calculate the gradient of the smoothed sea level pressure field

   slp_smooth_grad = calc_gradient.calc_gradient(slp_smooth, to_np(lats), to_np(lons))
   grad_mag = np.sqrt((slp_smooth_grad[0]**2.0) + (slp_smooth_grad[1]**2.0)).magnitude

# Automatically calculate the border region (in which high and low markers cannot be added) and remove high terrain regions too

   edge_height = int(np.shape(slp)[0]/40)+1
   edge_width = int(np.shape(slp)[1]/40)+1

   mask = np.where(ter > 1000.0, 1, 0)
   mask[0:edge_height,:] = 1
   mask[-edge_height:,:] = 1
   mask[:,0:edge_width] = 1
   mask[:,-edge_width:] = 1
   grad_mag_masked = np.ma.masked_array(grad_mag, mask)
   slp_smooth_masked = np.ma.masked_array(slp_smooth, mask)

# Calculate the slp gradient threshold to identify regions that are maxima and minima (and saddle points) This is set as the lowest 10% of the total slp gradient range. Create a binary image of regions that meet this criteria

   grad_low_thresh = np.ma.min(grad_mag_masked) + (np.ma.max(grad_mag_masked)-np.ma.min(grad_mag_masked))/5.0
   grad_high_thresh = np.ma.min(grad_mag_masked) + (np.ma.max(grad_mag_masked)-np.ma.min(grad_mag_masked))/10.0
   grad_low_thresh_bin = np.ma.where(grad_mag_masked < grad_low_thresh, 1, 0)
   grad_high_thresh_bin = np.ma.where(grad_mag_masked < grad_high_thresh, 1, 0)

# Repeat thresholding for low and high slp using the top 5% and bottom 20% of the total slp range and create binary images of regions that meet the critera.

   slp_low_thresh = np.ma.min(slp_smooth_masked) + (np.ma.max(slp_smooth_masked)-np.ma.min(slp_smooth_masked))/ 5.0

   slp_high_thresh = np.ma.max(slp_smooth_masked) - (np.ma.max(slp_smooth_masked)-np.ma.min(slp_smooth_masked))/20.0

   slp_low_thresh_bin = np.ma.where(slp_smooth_masked < slp_low_thresh, 1, 0)
   slp_high_thresh_bin = np.ma.where(slp_smooth_masked > slp_high_thresh, 1, 0)

# create low binary for regions that meet both low pressure and low slp gradient conditions, loop over contigious regions that meet the criteria and identify the location (lat lon and indices) of the minimum pressure value. This also prevents repeated values and removes points that are withion 500 km of each other.

   low_bin = slp_low_thresh_bin & grad_low_thresh_bin
   low_bin_labelled = label(low_bin)
   low_lats = []
   low_lons = []
   low_id0 = []
   low_id1 = []

   for j in np.arange(1, np.ma.max(low_bin_labelled)+1, 1):
      low_temp_mask = np.where(low_bin_labelled == j, 0, 1)
      low_slp_temp_masked = np.ma.masked_array(slp_smooth_masked, low_temp_mask)
      low_indices = np.unravel_index(np.ma.argmin(low_slp_temp_masked), low_slp_temp_masked.shape)
      if (any(low_lats == lats[low_indices[0],low_indices[1]].values)):
         for k in [low_lats.index(lats[low_indices[0],low_indices[1]].values)]:
            if (low_lons[k] == lons[low_indices[0],low_indices[1]].values):
               print("Low pressure centres repeated, will not be entered")
            else:
               low_lats.append(lats[low_indices[0],low_indices[1]].values)
               low_lons.append(lons[low_indices[0],low_indices[1]].values)
               low_id0.append(low_indices[0])
               low_id1.append(low_indices[1])
      else:
         low_lats.append(lats[low_indices[0],low_indices[1]].values)
         low_lons.append(lons[low_indices[0],low_indices[1]].values)
         low_id0.append(low_indices[0])
         low_id1.append(low_indices[1])

      del low_temp_mask


   low_dist = {}
   for j in np.arange(1,np.size(low_lats),1):
      for k in np.arange(0,j,1):
         low_dist[str(j)+"_"+str(k)] = geod.inv(low_lons[j], low_lats[j], low_lons[k], low_lats[k])[2]/1000.0

   del_list = []
   for key in low_dist:
      if ( low_dist[key] < 500.0 ):
         id1 = key.split('_')[0]
         id2 = key.split('_')[1]
         if (slp[low_id0[int(id1)],low_id1[int(id1)]] < slp[low_id0[int(id2)],low_id1[int(id2)]] ):
            del_list.append(int(id2))
         else:
            del_list.append(int(id1))

   for id in sorted(set(del_list), reverse=True):
      del low_lats[id]
      del low_lons[id]
      del low_id0[id]
      del low_id1[id]

# Repeat the above step but for high pressure centres.

   high_bin = slp_high_thresh_bin & grad_high_thresh_bin
   high_bin_labelled = label(high_bin)
   high_lats = []
   high_lons = []
   high_id0 = []
   high_id1 = []

   for j in np.arange(0, np.ma.max(high_bin_labelled)+1, 1):
      high_temp_mask = np.where(high_bin_labelled == j, 0, 1)
      high_slp_temp_masked = np.ma.masked_array(slp_smooth_masked, high_temp_mask)
      high_indices = np.unravel_index(high_slp_temp_masked.argmax(), high_slp_temp_masked.shape)
      if (any(high_lats == lats[high_indices[0],high_indices[1]].values)):
         for k in [high_lats.index(lats[high_indices[0],high_indices[1]].values)]:
            if (high_lons[k] == lons[high_indices[0],high_indices[1]].values):
               print("High pressure centres repeated, will not be entered")
            else:
               high_lats.append(lats[high_indices[0],high_indices[1]].values)
               high_lons.append(lons[high_indices[0],high_indices[1]].values)
               high_id0.append(high_indices[0])
               high_id1.append(high_indices[1])
      else:
         high_lats.append(lats[high_indices[0],high_indices[1]].values)
         high_lons.append(lons[high_indices[0],high_indices[1]].values)
         high_id0.append(high_indices[0])
         high_id1.append(high_indices[1])

      del high_temp_mask

   high_dist = {}
   for j in np.arange(1,np.size(high_lats),1):
      for k in np.arange(0,j,1):
         high_dist[str(j)+"_"+str(k)] = geod.inv(high_lons[j], high_lats[j], high_lons[k], high_lats[k])[2]/1000.0

   del_list = []
   for key in high_dist:
      if ( high_dist[key] < 500.0 ):
         id1 = key.split('_')[0]
         id2 = key.split('_')[1]
         if (slp[high_id0[int(id1)],high_id1[int(id1)]] < slp[high_id0[int(id2)],high_id1[int(id2)]] ):
            del_list.append(int(id2))
         else:
            del_list.append(int(id1))

   for id in sorted(set(del_list), reverse=True):
      del high_lats[id]
      del high_lons[id]
      del high_id0[id]
      del high_id1[id]

# Create figure and axes
   fig = plt.figure(figsize=(10,10))
   ax = plt.axes(projection=cart_proj)
   ax.coastlines(linewidth=0.5)

# Plot sea-level presssure
   slp_lvl = np.arange(900.0, 1100.0, 4.0)

   slp_plot = plt.contour(lons, lats, slp_smooth, levels=slp_lvl, colors='black', transform=crs.PlateCarree())
   plt.clabel(slp_plot, inline=True, fontsize=10, fmt='%.0f')

# Identify whether domain is portrait or landscape

   if np.size(lats[:,0]) < np.size(lats[0,:]):
      portrait = True
   else:
      portrait = False

# Add inset title

   titlebox = inset_axes(ax, '5%', '90%', loc = 7)
   [titlebox.spines[k].set_visible(False) for k in titlebox.spines]
   titlebox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
   titlebox.set_facecolor([1,1,1,0.7])

   titlebox.text(0.5,0.5, "Mean Sea Level Pressure (hPa)", rotation=90.0, verticalalignment='center', horizontalalignment='center')

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

# Calculate height and width of domain in grid points and degrees latitude and longitude

   dom_h = np.max(lats) - np.min(lats)
   dom_w = np.max(lons) - np.min(lons)
   dim_w = np.shape(lats)[1]
   dim_h = np.shape(lats)[0]

# Add markers and high and low labels
   for j in np.arange(0,np.size(low_lons), 1):
      ax.plot(low_lons[j], low_lats[j], color='black', linewidth=2, marker='x', transform=crs.PlateCarree())
      if ( low_id1[j] > dim_w/2 ):
         x_offset = -1.0*(dom_w/20.0)
      else:
         x_offset = dom_w/20.0
      if ( low_id0[j] > dim_h/2 ):
         y_offset = -1.0*(dom_h/20.0)
      else:
         y_offset = dom_h/20.0

      ax.text(low_lons[j]+x_offset, low_lats[j]+y_offset, "L", fontsize=30, fontweight="bold", horizontalalignment='right', verticalalignment='bottom', color='black', transform=crs.PlateCarree())
      ax.text(low_lons[j]+x_offset, low_lats[j]+y_offset, "%.0f" % slp[low_id0[j],low_id1[j]], fontsize=12, fontweight="bold", horizontalalignment='left',  verticalalignment='top', color='black', transform=crs.PlateCarree())


   for j in np.arange(0,np.size(high_lons), 1):
      ax.plot(high_lons[j], high_lats[j], color='black', linewidth=2, marker='x', transform=crs.PlateCarree())
      if ( high_id1[j] > dim_w/2 ):
         x_offset = -1.0*(dom_w/20.0)
      else:
         x_offset = dom_w/20.0
      if ( high_id0[j] > dim_h/2 ):
         y_offset = -1.0*(dom_h/20.0)
      else:
         y_offset = dom_h/20.0

      ax.text(high_lons[j]+x_offset, high_lats[j]+y_offset, "H", fontsize=30, fontweight="bold", horizontalalignment='right', verticalalignment='bottom', color='black', transform=crs.PlateCarree())
      ax.text(high_lons[j]+x_offset, high_lats[j]+y_offset, "%.0f" % slp[high_id0[j],high_id1[j]], fontsize=12, fontweight="bold", horizontalalignment='left',  verticalalignment='top', color='black', transform=crs.PlateCarree())


# Save image

   grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']

   plt.savefig(dest_dir+"/mslp_d0"+str(grid_id)+"_"+sim_start_time['SIMULATION_START_DATE']+"_valid_"+valid_time[0:16]+".png", bbox_inches='tight')
   plt.close()
