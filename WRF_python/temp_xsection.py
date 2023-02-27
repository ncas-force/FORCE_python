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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Geod
from scipy import interpolate

from wrf import (getvar, interplevel, interpline, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs)

# Input points along which the cross section will be calculated, this can be any number of points as long as the number of lat and lon points match each other, this can be supplied as an argument in the future
# waypoints for flights will include altitudes so that a route will be marked on a cross section too.

xsection_lats = [51.47, 52.25, 52.35, 52.56, 52.70, 53.05, 53.16, 53.52, 53.70, 53.86]
xsection_lons = [-0.46, -0.85, -1.09, -1.16, -1.31, -1.67, -1.74, -1.96, -1.65, -1.66]
xsection_alts = [0.0, 3000.0, 5000.0, 7500.0, 8000.0, 8000.0, 7500.0, 5000.0, 3000.0, 0.0]

# Input base and top altitudes (can be supplied as arguments in future)

base_alt = 0.0
top_alt = 10000.0

# Input contour level min and max and step

cmin = -80.0
cmax = 20.0
cstep = 1.0

# Calculate levels to interpolate onto

lev_interval = (top_alt - base_alt)/100.0
interp_levs = np.arange(base_alt, top_alt+lev_interval, lev_interval)

# Calculate great circle distance between waypoints and create cross section locations based on waypoint lats and lons

geod = Geod(ellps='WGS84')

xsection_locs = []
gc_dist = []

for i in np.arange(0, np.size(xsection_lats), 1):
   xsection_locs.append(CoordPair(lat=xsection_lats[i], lon=xsection_lons[i]))
   if i > 0:
      gc_dist.append(geod.inv(xsection_lons[i-1], xsection_lats[i-1], xsection_lons[i], xsection_lats[i])[2]/1000.0)

# Calculate total length of cross section and percentage of each section (including integer version for plot scaling)

total_dist = np.sum(gc_dist)
dist_percent = 100.0*(gc_dist/total_dist)
dist_percent_int = []
for i in np.arange(0, np.size(dist_percent), 1):
   dist_percent_int.append(int(dist_percent[i]))

# Input WRF out file (this will be given as an argument when running operationally
wrf_fil ="../../example_WRF_output/wrfout_d02_2022-10-31_18:00:00"

# Check for existance of WRF out file
if not os.path.exists(wrf_fil):
    raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
wrf_in= Dataset(wrf_fil)

# Extract the number of times within the WRF file and loop over all times in file
num_times = np.size(extract_times(wrf_in, ALL_TIMES))

#for i in np.arange(0, num_times, 1):
for i in np.arange(0, 1, 1):

# Read height, temperature and terrain
   ht = getvar(wrf_in, 'z', timeidx=i)
   t = getvar(wrf_in, 'tc', timeidx=i)
   ter = getvar(wrf_in, "ter", timeidx=-1)

# Compute vertical cross section interpolation

   t_cross = []
   ter_cross = []

   for j in np.arange(0, np.size(gc_dist), 1):
      t_cross.append(vertcross(t, ht, wrfin=wrf_in, levels=interp_levs, start_point=xsection_locs[j], end_point=xsection_locs[j+1], latlon=True, meta=True))
      ter_cross.append(interpline(ter, wrfin=wrf_in, start_point=xsection_locs[j], end_point=xsection_locs[j+1]))

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) 
   cart_proj = get_cartopy(t)
   lats, lons = latlon_coords(t)

# Create figure and axes remove wspace and hspace to make panels butt against each other.
   fig = plt.figure(figsize=(15,5))
   plt.subplots_adjust(wspace=0, hspace=0)

# create gridspec for subplot scaling
   gs = fig.add_gridspec(1,np.size(gc_dist), width_ratios=dist_percent_int)

# Set contour levels
   t_lvls = np.arange(cmin, cmax, cstep)
   t_lvls2 = np.arange(cmin, cmax, 2.0*cstep)

# loop through parts of cross section and plot data
   for j in np.arange(0, np.size(gc_dist), 1):
      ax = fig.add_subplot(gs[j])
      ax.axis(ymin=0.0, ymax=100.0)
      vert_vals = to_np(t_cross[j].coords["vertical"])
      v_ticks = np.arange(vert_vals.shape[0])
      thin = [int(x/10.0) for x in v_ticks.shape]
      ax.set_yticks(v_ticks[::thin[0]])
      x_ticks = np.arange(0.0, np.shape(t_cross[j])[1], 1)
      t_cross_np = np.where(np.isnan(to_np(t_cross[j])), np.nan ,to_np(t_cross[j]))

# Fill in nans at the bottom of the cross section with nearest value from above (this might be a bit of a bodge and might not be the fastest way to do it).
      for k in np.arange(0, np.size(x_ticks), 1):
         ind = np.where(~np.isnan(t_cross_np[:,k]))[0]
         first, last = ind[0], ind[-1]
         t_cross_np[:first,k] = t_cross_np[first,k]
         t_cross_np[last + 1:,k] = t_cross_np[last,k]

# Create contours 
      t_contours = ax.contourf(x_ticks, v_ticks, t_cross_np, levels=t_lvls, cmap='jet')
      t_contours2 = ax.contour(x_ticks, v_ticks, t_cross_np, levels=t_lvls2, colors='k')

# correct waypoint altitudes so that they cannot be below the suface

      if xsection_alts[j] < to_np(ter_cross[j])[0]:
         xsection_alts1 = to_np(ter_cross[j])[0]
      else:
         xsection_alts1 = xsection_alts[j]

      if xsection_alts[j+1] < to_np(ter_cross[j])[-1]:
         xsection_alts2 = to_np(ter_cross[j])[-1]
      else:
         xsection_alts2 = xsection_alts[j+1]

# Plot polyline indicating path of aircraft through the cross section
      ax.plot([0, x_ticks[-1]], [100.0*((xsection_alts1-base_alt)/(top_alt-base_alt)), 100.0*((xsection_alts2-base_alt)/(top_alt-base_alt))], color='red')

# Plot terrain at the bottom of the cross section
      ht_fill = ax.fill_between(x_ticks, 0, 100.0*((to_np(ter_cross[j])-base_alt)/(top_alt-base_alt)), facecolor="black")

# Setup tickmarks so that only the first panel has y axis labels and x axis labels are lat and lon values      
      if j == 0 :
         ax.set_yticklabels(vert_vals[::thin[0]], fontsize=10)
         ax.set_xticks([x_ticks[0], x_ticks[-1]])
         ax.set_xticklabels([str(xsection_lats[j])+"\n"+str(xsection_lons[j]), str(xsection_lats[j+1])+"\n"+str(xsection_lons[j+1])], fontsize=10)
      else:
         ax.set_yticklabels([])
         ax.set_xticks([x_ticks[-1]])
         ax.set_xticklabels([str(xsection_lats[j+1])+"\n"+str(xsection_lons[j+1])], fontsize=10)

# Set position of contour labels so that they only occur in panels that are greater than 15 % of the total width and occur down tha middle of the panel
      if dist_percent_int[j] >= 15:
         mid_x = float(x_ticks[-1])/2.0
         clab_locs = []
         for line in t_contours2.collections:
            for path in line.get_paths():
                verts = path.vertices
              
                print(mid_x)
                print(verts[:,0])
                print(verts[np.abs(mid_x-verts[:,0]).argmin(),0])
                x_val = verts[np.abs(mid_x-verts[:,0]).argmin(),0]
                if x_val == 0 or x_val == x_ticks[-1]:
                   x_val = mid_x
                
                clab_locs.append([x_val,np.mean(verts[:,1])])

         ax.clabel(t_contours2, fontsize=9, inline=1, fmt='%2.0f', manual=clab_locs)

# Save figure
   plt.savefig('crosssection_test.png', bbox_inches='tight')

