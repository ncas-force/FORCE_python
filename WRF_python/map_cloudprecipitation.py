def map_cloudprecipitation(x):
   
   import numpy as np
   from cartopy import crs, feature
   from cartopy.feature import NaturalEarthFeature
   import matplotlib.pyplot as plt
   import matplotlib as mpl
   from netCDF4 import Dataset
   import os
   from mpl_toolkits.axes_grid1.inset_locator import inset_axes
   from datetime import datetime, timedelta

   from wrf import (getvar, interplevel, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs, ll_to_xy, get_proj_params, getproj)

# Read spatial information from input dictionary

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
      x["locationname"] = " "
   map_name = x["locationname"]

# Input WRF out file as an argument (full path)
   wrf_fil = x["infile"]

   path = os.path.dirname(wrf_fil)
   wrf_fil_head = os.path.basename(wrf_fil)[0:11]

   YYYY = os.path.basename(wrf_fil)[11:15]
   MM = os.path.basename(wrf_fil)[16:18]
   DD = os.path.basename(wrf_fil)[19:21]
   hh = os.path.basename(wrf_fil)[22:24]
   mm = os.path.basename(wrf_fil)[25:27]
   ss = os.path.basename(wrf_fil)[28:30]

   wrf_fil2 =path+"/"+wrf_fil_head+(datetime(int(YYYY), int(MM), int(DD), int(hh), int(mm), int(ss))-timedelta(hours=1)).strftime("%Y-%m-%d_%H:%M:%S")

# Check for existance of WRF out file
   if not os.path.exists(wrf_fil):
      raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
   wrf_in = Dataset(wrf_fil)
   proj = getproj(**get_proj_params(wrf_in))
   projection = proj.cf()['grid_mapping_name']

# Extract the number of times within the WRF file and loop over all times in file
   num_times = np.size(extract_times(wrf_in, ALL_TIMES))

   for i in np.arange(0, num_times, 1):

      cloud_frac_all = getvar(wrf_in, 'cloudfrac', timeidx=i)[0,:,:]

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting)
      cart_proj = get_cartopy(cloud_frac_all)
      lats_all, lons_all = latlon_coords(cloud_frac_all)

# Check that the supplied lat and lon values are within the WRF domain, if not then the plot will not be created.
      if limit_id == "all":
         x1_y1 = (1, 1)
         x2_y2 = (1, 1)
      else:
         x1_y1 = ll_to_xy(wrf_in, limit_lats[0], limit_lons[0])
         x2_y2 = ll_to_xy(wrf_in, limit_lats[1], limit_lons[1])

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) 
      if x1_y1[1] >= 0 and x1_y1[1] < np.shape(cloud_frac_all)[0] and x1_y1[0] >= 0 and x1_y1[0] < np.shape(cloud_frac_all)[1] and x2_y2[1] >= 0 and x2_y2[1] < np.shape(cloud_frac_all)[0] and x2_y2[0] >= 0 and x2_y2[0] < np.shape(cloud_frac_all)[1]:

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

# Subset latitudes and longitudes 
         lats = lats_all[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         lons = lons_all[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]


# Read in and calculate cloud fraction and precipitation amounts

         cloud_frac = getvar(wrf_in, 'cloudfrac', timeidx=i)[:,min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         cloud_frac_max = np.amax(cloud_frac, axis=0)

         tc = getvar(wrf_in, 'tc', timeidx=i)[:,min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         pressure = getvar(wrf_in, 'pressure', timeidx=i)[:,min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]

         tc_850 = interplevel(tc, pressure, 850.0)

         if i == 0:

            if os.path.exists(wrf_fil2):

               wrf_in2 = Dataset(wrf_fil2)
               num_times2 = np.size(extract_times(wrf_in2, ALL_TIMES))

               rainc1 = getvar(wrf_in, 'RAINC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
               rainnc1 = getvar(wrf_in, 'RAINNC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
               snownc1 = getvar(wrf_in, 'SNOWNC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
               hailnc1 = getvar(wrf_in, 'HAILNC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
               graupel1 = getvar(wrf_in, 'GRAUPELNC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]

               rainc2 = getvar(wrf_in2, 'RAINC', timeidx=num_times2-1)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
               rainnc2 = getvar(wrf_in2, 'RAINNC', timeidx=num_times2-1)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
               snownc2 = getvar(wrf_in2, 'SNOWNC', timeidx=num_times2-1)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
               hailnc2 = getvar(wrf_in2, 'HAILNC', timeidx=num_times2-1)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
               graupel2 = getvar(wrf_in2, 'GRAUPELNC', timeidx=num_times2-1)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]

            else:
               rainc1 = getvar(wrf_in, 'RAINC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
               rainnc1 = getvar(wrf_in, 'RAINNC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
               snownc1 = getvar(wrf_in, 'SNOWNC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
               hailnc1 = getvar(wrf_in, 'HAILNC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
               graupel1 = getvar(wrf_in, 'GRAUPELNC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]

               rainc2 = rainc1 * 0.0
               rainnc2 = rainnc1 * 0.0
               snownc2 = snownc1 * 0.0
               hailnc2 = hailnc1 * 0.0
               graupel2 = graupel1 * 0.0

         else:
            rainc1 = getvar(wrf_in, 'RAINC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
            rainnc1 = getvar(wrf_in, 'RAINNC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
            snownc1 = getvar(wrf_in, 'SNOWNC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
            hailnc1 = getvar(wrf_in, 'HAILNC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
            graupel1 = getvar(wrf_in, 'GRAUPELNC', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]

            rainc2 = getvar(wrf_in, 'RAINC', timeidx=i-1)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
            rainnc2 = getvar(wrf_in, 'RAINNC', timeidx=i-1)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
            snownc2 = getvar(wrf_in, 'SNOWNC', timeidx=i-1)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
            hailnc2 = getvar(wrf_in, 'HAILNC', timeidx=i-1)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
            graupel2 = getvar(wrf_in, 'GRAUPELNC', timeidx=i-1)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]

         rtot1hr = rainc1 + rainnc1 + snownc1 + hailnc1 + graupel1 - rainc2 - rainnc2 - snownc2 - hailnc2 - graupel2
 
         tc_850 = np.where(rtot1hr <= 0.25, 0.0, tc_850)

# Create figure and axes
         fig = plt.figure(figsize=(10,10))
         ax = plt.axes(projection=cart_proj)
         ax.coastlines(linewidth=0.5)
         ax.add_feature(feature.OCEAN,facecolor=("lightblue"))
         ax.add_feature(feature.LAND,facecolor=("sandybrown"))
         gl = ax.gridlines(linewidth=0.5, draw_labels=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
         gl.right_labels = False
         gl.bottom_labels = False
         
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

         precip_lvls = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 40.0, 50.0]
         cmap = mpl.cm.get_cmap('jet')
         cmap_sub = cmap(np.linspace(0.25,0.9, 22))
         plt.contourf(lons, lats, rtot1hr, levels=precip_lvls, colors=cmap_sub,  zorder=2, transform=crs.PlateCarree())

         if np.size(lats[:,0]) < np.size(lats[0,:]):
            portrait = True
         else:
            portrait = False

# Create inset colourbar

         cbar_inset = False

         if cbar_inset:

            if portrait:
               cbbox = inset_axes(ax, '13%', '90%', loc = 7)
               [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
               cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
               cbbox.set_facecolor([1,1,1,0.7])
               cbbox.text(0.82,0.5, "1 hour accumulated precipitation (mm)", rotation=90.0, verticalalignment='center', horizontalalignment='center')
               cbbox.text(0.9,0.25, "Cloud cover (white background shading)", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='black')
               cbbox.text(0.9,0.75, u'$\u00D7$'+" Chance of snow", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='black')
               cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
               cb = plt.colorbar(cax=cbaxes, aspect=20)
            else:
               cbbox = inset_axes(ax, '90%', '12%', loc = 8)
               [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
               cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
               cbbox.set_facecolor([1,1,1,0.7])
               cbbox.text(0.5,0.28, "1 hour accumulated precipitation (mm)", verticalalignment='center', horizontalalignment='center')
               cbbox.text(0.75,0.1, u'$\u00D7$'+" Chance of snow", verticalalignment='center', horizontalalignment='center', color='black')
               cbbox.text(0.25,0.1, "Cloud cover (white background shading)", verticalalignment='center', horizontalalignment='center', color='black')
               cbaxes = inset_axes(cbbox, '95%', '30%', loc = 9)
               cb = plt.colorbar(cax=cbaxes, orientation='horizontal')
         else:
            cbbox = inset_axes(ax, '100%', '100%', bbox_to_anchor=(0, -0.13, 1, 0.13), bbox_transform=ax.transAxes, loc = 8, borderpad=0)
            [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
            cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
            cbbox.set_facecolor([1,1,1,0.7])
            cbbox.text(0.5,0.3, "1 hour accumulated precipitation (mm)", verticalalignment='center', horizontalalignment='center')
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

            grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']

# Return figure
            return(fig)

###############################################################################################################################################################################################
# If the script is called as the main script then this part of the code is exectued first. The plotting section above can be called seperately as a module using a dictionary as the only input. 

if __name__ == "__main__":

   import os
   import sys
   import csv
   import numpy as np
   import matplotlib.pyplot as plt

# Define destination directory
   dest_dir = "/home/earajr/FORCE_WRF_plotting/output/cloudprecipitation"
   if not os.path.isdir(dest_dir):
       os.makedirs(dest_dir)

# Define input directory
   input_dir = "/home/earajr/FORCE_WRF_plotting/WRF_plot_inputs"

   limit_lats = []
   limit_lons = []
   map_names = []

   with open(input_dir+"/map_limit_lats", "r") as file:
       reader = csv.reader(file)
       for row in reader:
           limit_lats.append(row)

   with open(input_dir+"/map_limit_lons", "r") as file:
       reader = csv.reader(file)
       for row in reader:
           limit_lons.append(row)

   with open(input_dir+"/map_names", "r") as file:
       reader = csv.reader(file)
       for row in reader:
           map_names.append(row)

   if (np.shape(limit_lats)[0] == np.shape(limit_lons)[0] == np.size(map_names)):
      print("Number of map limit latitudes, longitudes and map names is correct continuing with map generation.")
   else:
      raise ValueError("The number of map limit latitudes, longitudes or map names in the input directory does not match, please check that the map information provided is correct")

# Input WRF out file as an argument (full path)
   wrf_fil = sys.argv[1]

# Loop through maps, create input dictionary for each map and pass it to the map_cloudprecipitation function above

   for i in np.arange(0, np.shape(limit_lats)[0], 1):

      input_dict = {}
      input_dict["latitudes"] = limit_lats[i]
      input_dict["longitudes"] = limit_lons[i]
      input_dict["infile"] = wrf_fil
      input_dict["locationname"] = map_names[i]

      fig = map_cloudprecipitation(input_dict)

      plt.savefig(dest_dir+"/cloudprecipitationtest_"+map_names[i][0]+".png", bbox_inches='tight')


## Define destination direction
#dest_dir = "/home/earajr/FORCE_WRF_plotting/output/cloudprecip"
#if not os.path.isdir(dest_dir):
#    os.makedirs(dest_dir)
#
## Input WRF out file as an argument (full path)
#wrf_fil = sys.argv[1]
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
#if not os.path.exists(wrf_fil):
#    raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
#wrf_in= Dataset(wrf_fil)
#times = extract_times(wrf_in, ALL_TIMES)
#
## Extract the number of times within the WRF file and loop over all times in file
#num_times = np.size(extract_times(wrf_in, ALL_TIMES))
#
#for i in np.arange(0, num_times, 1):
#
#   cloud_frac = getvar(wrf_in, 'cloudfrac', timeidx=i)
#   cloud_frac_max = np.amax(cloud_frac, axis=0)
#
#   tc = getvar(wrf_in, 'tc', timeidx=i)
#   pressure = getvar(wrf_in, 'pressure', timeidx=i)
#
#   tc_850 = interplevel(tc, pressure, 850.0)
#
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
#
## Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) 
#   cart_proj = get_cartopy(rainc1)
#   lats, lons = latlon_coords(rainc1)

## Create figure and axes
#   fig = plt.figure(figsize=(10,10))
#   ax = plt.axes(projection=cart_proj)
#   ax.coastlines(linewidth=0.5)
#   ax.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
#   ax.add_feature(cartopy.feature.LAND,facecolor=("sandybrown"))
#
### Plot cloud cover
#
#   cloud_lvls = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
#   color1 = [1,1,1,0]
#   color2 = [1,1,1,1]
#   cloud_cmap = np.linspace(color1, color2, 16)
#   cloud = plt.contourf(lons, lats, cloud_frac_max, levels=cloud_lvls, colors=cloud_cmap, zorder=1, antialiased=True, transform=crs.PlateCarree(), extend='max')
#
## Plot snow hatching
#
#   snow_lvls = [-10.0, -5.0, 0.0]
#   snow = plt.contourf(lons, lats, tc_850, levels=snow_lvls, colors='None', hatches=['XX','',''], zorder=3, transform=crs.PlateCarree())
#
## Plot precip
#
#   precip_lvls = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 40.0, 50.0]
#   cmap = mpl.cm.get_cmap('jet')
#   cmap_sub = cmap(np.linspace(0.25,0.9, 22))
#   plt.contourf(lons, lats, rtot1hr, levels=precip_lvls, colors=cmap_sub,  zorder=2, transform=crs.PlateCarree())
#
#   if np.size(lats[:,0]) < np.size(lats[0,:]):
#      portrait = True
#   else:
#      portrait = False
#
## Create inset colourbar
#
#   if portrait:
#      cbbox = inset_axes(ax, '13%', '90%', loc = 7)
#      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
#      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
#      cbbox.set_facecolor([1,1,1,0.7])
#      cbbox.text(0.7,0.5, "1 hour accumulated precipitation (mm)", rotation=90.0, verticalalignment='center', horizontalalignment='center')
#      cbbox.text(0.85,0.25, "Cloud cover (white background shading)", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='black')
#      cbbox.text(0.85,0.75, u'$\u00D7$'+" Chance of snow", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='black')
#      cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
#      cb = plt.colorbar(cax=cbaxes, aspect=20)
#   else:
#      cbbox = inset_axes(ax, '90%', '12%', loc = 8)
#      [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
#      cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
#      cbbox.set_facecolor([1,1,1,0.7])
#      cbbox.text(0.5,0.3, "1 hour accumulated precipitation (mm)", verticalalignment='center', horizontalalignment='center')
#      cbbox.text(0.75,0.15, u'$\u00D7$'+" Chance of snow", verticalalignment='center', horizontalalignment='center', color='black')
#      cbbox.text(0.25,0.15, "Cloud cover (white background shading)", verticalalignment='center', horizontalalignment='center', color='black')
#      cbaxes = inset_axes(cbbox, '95%', '30%', loc = 9)
#      cb = plt.colorbar(cax=cbaxes, orientation='horizontal')
#
## Add inset timestamp
#   tsbox = inset_axes(ax, '95%', '3%', loc = 9)
#   [tsbox.spines[k].set_visible(False) for k in tsbox.spines]
#   tsbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
#   tsbox.set_facecolor([1,1,1,1])
#   
#   sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')
#   valid_time = str(extract_times(wrf_in, ALL_TIMES)[i])[0:22]
#
##   print(sim_start_time['SIMULATION_START_DATE'])
#
#   tsbox.text(0.01, 0.45, "Start date: "+sim_start_time['SIMULATION_START_DATE'], verticalalignment='center', horizontalalignment='left')
#   tsbox.text(0.99, 0.45, "Valid_date: "+valid_time, verticalalignment='center', horizontalalignment='right')
#
#   grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']
#   
#   plt.savefig(dest_dir+"/cloudprecip_d0"+str(grid_id)+"_"+sim_start_time['SIMULATION_START_DATE']+"_valid_"+valid_time[0:16]+".png", bbox_inches='tight')
#   plt.close()

