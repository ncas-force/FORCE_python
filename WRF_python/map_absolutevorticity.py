def map_absolutevorticity(x):
   import numpy as np
   from cartopy import crs
   import matplotlib.pyplot as plt
   import matplotlib as mpl
   import matplotlib.colors as mcolors
   import colorcet as cc
   from netCDF4 import Dataset
   import os
   from mpl_toolkits.axes_grid1.inset_locator import inset_axes
   from pyproj import Geod

   from wrf import (getvar, interplevel, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs, ll_to_xy, get_proj_params, getproj)

# Read spatial information from input directory

   limit_lats = [float(x["latitudes"][0]), float(x["latitudes"][1])]
   limit_lons = [float(x["longitudes"][0]), float(x["longitudes"][1])]

# define whether map corners are ullr or llur and switch values so that leftmost point is always at index 0
   if (limit_lats[0] > limit_lats[1] and limit_lons[0] < limit_lons[1]):
      limit_id = "ullr"
   elif (limit_lats[0] < limit_lats[1] and limit_lons[0] > limit_lons[1]):
      limit_id = "ullr"
      temp_lat = limit_lats[0]
      limit_lats[0] = limit_lats[1]
      limit_lats[1] = temp_lat

      temp_lon = limit_lons[0]
      limit_lons[0] = limit_lons[1]
      limit_lons[1] = temp_lon
   elif (limit_lats[0] > limit_lats[1] and limit_lons[0] > limit_lons[1]):
      limit_id = "llur"
      temp_lat = limit_lats[0]
      limit_lats[0] = limit_lats[1]
      limit_lats[1] = temp_lat

      temp_lon = limit_lons[0]
      limit_lons[0] = limit_lons[1]
      limit_lons[1] = temp_lon
   elif (limit_lats[0] < limit_lats[1] and limit_lons[0] < limit_lons[1]):
      limit_id = "llur"
   else:
      limit_id = "all"

# set the name of the map
   if "locationname" not in x:
      x["locationname"] = [" "]
   map_name = x["locationname"][0]

# read level type
   if "leveltype" not in x:
      x["leveltype"] = ["pressure"]
   leveltype = x["leveltype"]

# read in levels to be plotted

   if "levels" not in x:
      x["levels"] = [925]
   levels = float(x["levels"][0])

# Input WRF out file as an argument (full path)
   wrf_fil = x["infile"]

# Check for existance of WRF out file
   if not os.path.exists(wrf_fil):
      raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf and read projection of WRF simulations
   wrf_in = Dataset(wrf_fil)
   proj = getproj(**get_proj_params(wrf_in))
   projection = proj.cf()['grid_mapping_name']

# Define geodesy
   geod = Geod(ellps='WGS84')

# Extract the number of times within the WRF file and loop over all times in file
   num_times = np.size(extract_times(wrf_in, ALL_TIMES))

# Loop over times present in input file
   for i in np.arange(0, num_times, 1):

# Read in grid and time inforation
      grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']
      sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')
      valid_time = str(extract_times(wrf_in, ALL_TIMES)[i])[0:22]

# Read all sea level pressure
      slp_all = getvar(wrf_in, 'slp', timeidx=i)[:,:]

# Check that the supplied lat and lon values are within the WRF domain, if not then the plot will not be created.
      if limit_id == "all":
         x1_y1 = (1, 1)
         x2_y2 = (1, 1)
      else:
         x1_y1 = ll_to_xy(wrf_in, limit_lats[0], limit_lons[0])
         x2_y2 = ll_to_xy(wrf_in, limit_lats[1], limit_lons[1])

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) also identify the required max and min indices
      if x1_y1[1] >= 0 and x1_y1[1] < np.shape(slp_all)[0] and x1_y1[0] >= 0 and x1_y1[0] < np.shape(slp_all)[1] and x2_y2[1] >= 0 and x2_y2[1] < np.shape(slp_all)[0] and x2_y2[0] >= 0 and x2_y2[0] < np.shape(slp_all)[1]:
         cart_proj = get_cartopy(slp_all)
         lats_all, lons_all = latlon_coords(slp_all)

         if limit_id == "ullr":
            lat_lon_mask = np.where(lats_all < limit_lats[0], 1, 0) & np.where(lats_all > limit_lats[1], 1, 0) & np.where(lons_all > limit_lons[0], 1, 0) & np.where(lons_all < limit_lons[1], 1, 0)
         elif limit_id == "llur":
            lat_lon_mask = np.where(lats_all > limit_lats[0], 1, 0) & np.where(lats_all < limit_lats[1], 1, 0) & np.where(lons_all > limit_lons[0], 1, 0) & np.where(lons_all < limit_lons[1], 1, 0)
         else:
            lat_lon_mask = np.ones_like(lats_all)

         max_lat_ind = 0
         min_lat_ind = 10000

         max_lon_ind = 0
         min_lon_ind = 10000

         for j, k in zip(*np.where(lat_lon_mask)):
            if j > max_lat_ind:
               max_lat_ind = j
            if k > max_lon_ind:
               max_lon_ind = k
            if j < min_lat_ind:
               min_lat_ind = j
            if k < min_lon_ind:
               min_lon_ind = k

         if leveltype == "pressure":
            pres = getvar(wrf_in, 'pressure', timeidx=i)[:,min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         elif leveltype == "altitude":
            z = getvar(wrf_in, 'z', timeidx=i)[:,min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         else:
            raise ValueError("The provided level type does not match recognised values. Level type should be either 'pressure' or 'altitude'.")
            
         avo = getvar(wrf_in, 'avo', timeidx=i)[:,min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]

# Subset lat and lon from lats_all and lons_all
         lats = lats_all[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         lons = lons_all[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]

# Interpolate to level on correct level type

         if leveltype == "pressure":
            avo_level = interplevel(avo, pres, levels)
         if leveltype == "altitude":
            avo_level = interplevel(avo, z, levels)

# Create figure and axes
         fig = plt.figure(figsize=(10,10))
         ax = plt.axes(projection=cart_proj)
         ax.coastlines(linewidth=1.0)
         gl = ax.gridlines(linewidth=0.5, draw_labels=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
         gl.right_labels = False
         gl.bottom_labels = False

# Plot absolute vorticity

         avo_lvl = np.arange(-100.0, 105.0, 5.0)
         plt.contourf(lons, lats, avo_level, levels=avo_lvl, zorder=1, cmap=cc.cm['diverging_bwr_20_95_c54'], transform=crs.PlateCarree(), extend="both")

# Identify whether domain is portrait or landscape

         if np.size(lats[:,0]) < np.size(lats[0,:]):
            portrait = True
         else:
            portrait = False

# Create inset colourbar

         if leveltype == "pressure":
            colorbartext= "%.0f" %levels +" hPa absolute vorticity ($\mathregular{10^{-5}s^{-1}}$)"
         if leveltype == "altitude":
            colorbartext= "%.0f" %levels +" m absolute vorticity ($\mathregular{10^{-5}s^{-1}}$)"

         cbar_inset = False

         if cbar_inset:

            if portrait:
               cbbox = inset_axes(ax, '13%', '90%', loc = 7)
               [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
               cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
               cbbox.set_facecolor([1,1,1,0.7])
               cbbox.text(0.85,0.5, colorbartext, rotation=90.0, verticalalignment='center', horizontalalignment='center')
               cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
               cb = plt.colorbar(cax=cbaxes, aspect=20)
            else:
               cbbox = inset_axes(ax, '90%', '12%', loc = 8)
               [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
               cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
               cbbox.set_facecolor([1,1,1,0.7])
               cbbox.text(0.5,0.3, colorbartext, verticalalignment='center', horizontalalignment='center')
               cbaxes = inset_axes(cbbox, '95%', '30%', loc = 9)
               cb = plt.colorbar(cax=cbaxes, orientation='horizontal')
         else:
            cbbox = inset_axes(ax, '100%', '100%', bbox_to_anchor=(0, -0.13, 1, 0.13), bbox_transform=ax.transAxes, loc = 8, borderpad=0)
            [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
            cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
            cbbox.set_facecolor([1,1,1,0.7])
            cbbox.text(0.5,0.1, colorbartext, verticalalignment='center', horizontalalignment='center')
            cbaxes = inset_axes(cbbox, '95%', '30%', loc = 9)
            cb = plt.colorbar(cax=cbaxes, orientation='horizontal')
            

# Add inset timestamp
         tsbox = inset_axes(ax, '95%', '3%', loc = 9)
         [tsbox.spines[k].set_visible(False) for k in tsbox.spines]
         tsbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
         tsbox.set_facecolor([1,1,1,1])

         sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')
         valid_time = str(extract_times(wrf_in, ALL_TIMES)[i])[0:22]
   
#         print(sim_start_time['SIMULATION_START_DATE'])

         tsbox.text(0.01, 0.45, "Start date: "+sim_start_time['SIMULATION_START_DATE'], verticalalignment='center', horizontalalignment='left')
         tsbox.text(0.99, 0.45, "Valid_date: "+valid_time, verticalalignment='center', horizontalalignment='right')

# Return figure
         return(fig)

      else:
         print("Charts can only be generated for regions that are inside the WRF domain")


###############################################################################################################################################################################################
# If the script is called as the main script then this part of the code is exectued first. The plotting section above can be called seperately as a module using a dictionary as the only input.

if __name__ == "__main__":

   import os
   import sys
   import csv
   import numpy as np
   import matplotlib.pyplot as plt

# Define destination directory
#   dest_dir = "/home/earajr/FORCE_WRF_plotting/output/absolutevorticitylevels"
   dest_dir = sys.argv[2]
   if not os.path.isdir(dest_dir):
       os.makedirs(dest_dir)

## Define input directory
#   input_dir = "/home/earajr/FORCE_WRF_plotting/WRF_plot_inputs"
#
   limit_lats = []
   limit_lons = []
   map_names = []
   map_leveltype = []
   map_level = []
#
# Input WRF out file as an argument (full path)
   wrf_fil = sys.argv[1]
   base_wrf_fil = os.path.basename(wrf_fil)
   dom = base_wrf_fil.split("_")[1]
   date = base_wrf_fil.split("_")[2]
   time = base_wrf_fil.split("_")[3].replace(":", "-")

# Input level information as an argument

   level_flag = sys.argv[3]
   leveltype_flag = level_flag[0]
   if leveltype_flag == "p":
      map_leveltype = "pressure"
   if leveltype_flag == "a":
      map_leveltype = "altitude"
   map_level.append(level_flag[1:])

# Input map information

   map_names.append(sys.argv[4])
   limit_lats.append(sys.argv[5])
   limit_lats.append(sys.argv[6])
   limit_lons.append(sys.argv[7])
   limit_lons.append(sys.argv[8])

# Create input dictionary for each map and pass it to the map_absolutevorticity function above

   input_dict = {}
   input_dict["latitudes"] = limit_lats
   input_dict["longitudes"] = limit_lons
   input_dict["infile"] = wrf_fil
   input_dict["locationname"] = map_names
   input_dict["leveltype"] = map_leveltype
   input_dict["levels"] = map_level

   fig = map_absolutevorticity(input_dict)

   plt.savefig(dest_dir+"/absolutevorticity_"+dom+"_"+date+"_"+time+"_"+level_flag+"_"+map_names[0]+".png", bbox_inches='tight')

