def map_radiationfog(x):

   import numpy as np
   from cartopy import crs, feature
   from cartopy.feature import NaturalEarthFeature
   import matplotlib.pyplot as plt
   from matplotlib.colors import ListedColormap
   from netCDF4 import Dataset
   import os
#import sys
   from mpl_toolkits.axes_grid1.inset_locator import inset_axes
   from datetime import datetime, timedelta
   import calc_gradient

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

# Define secondary WRF input file (at 12 UTC)
   for i in np.arange(0, 24, 1):
      if (datetime(int(YYYY), int(MM), int(DD), int(hh))-timedelta(hours=int(i))).strftime("%H") == "12" :
         wrf_fil2 =path+"/"+wrf_fil_head+(datetime(int(YYYY), int(MM), int(DD), int(hh), int(mm), int(ss))-timedelta(hours=int(i))).strftime("%Y-%m-%d_%H:%M:%S")

# Check for existance of WRF out file
   if not os.path.exists(wrf_fil):
      raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Check for existance of WRF out file
   if not os.path.exists(wrf_fil2):
      raise ValueError("Warning! "+wrf_fil2+" does not exist, cannot calculate fog risk.")

# Read WRF out netcdf
   wrf_in = Dataset(wrf_fil)
   proj = getproj(**get_proj_params(wrf_in))
   projection = proj.cf()['grid_mapping_name']

# Read second WRF input file and identify the index where time is equal to 12
   wrf_in2= Dataset(wrf_fil2)
   times2 = extract_times(wrf_in2, ALL_TIMES)
   num_times2 = np.size(extract_times(wrf_in2, ALL_TIMES))

   for i in np.arange(0, num_times2, 1):
      if (times2[i].astype(str)[11:13]) == "12" and (times2[i].astype(str)[14:16]) == "00" and (times2[i].astype(str)[17:19]) == "00":
         noon_index = i

# Read all 2m dewpoint temperatures at 12 UTC (this will be used to define the projection, the lats_all, lons_all and identify if provided points are within the WRF domain.
   td2_12_all = getvar(wrf_in2, 'td2', timeidx=noon_index, units='degC')

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting)
   cart_proj = get_cartopy(td2_12_all)
   lats_all, lons_all = latlon_coords(td2_12_all)

# Check that the supplied lat and lon values are within the WRF domain, if not then the plot will not be created.
   if limit_id == "all":
      x1_y1 = (1, 1)
      x2_y2 = (1, 1)
   else:
      x1_y1 = ll_to_xy(wrf_in, limit_lats[0], limit_lons[0])
      x2_y2 = ll_to_xy(wrf_in, limit_lats[1], limit_lons[1])

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) 
   if x1_y1[1] >= 0 and x1_y1[1] < np.shape(td2_12_all)[0] and x1_y1[0] >= 0 and x1_y1[0] < np.shape(td2_12_all)[1] and x2_y2[1] >= 0 and x2_y2[1] < np.shape(td2_12_all)[0] and x2_y2[0] >= 0 and x2_y2[0] < np.shape(td2_12_all)[1]:

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

# Read in and calculate values on subsetted region designated by 
# Read in the 2m temperature at 12 UTC and convert to celcius
      t2_12 = getvar(wrf_in2, 'T2', timeidx=noon_index)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
      t2_12 = t2_12 - 273.15

# Read in the 2m dewpoint temperature at 12 UTC
      td2_12 = getvar(wrf_in2, 'td2', timeidx=noon_index, units='degC')[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]

# Calculate Y value from Craddock and Pritchard radiation fog method.
      Y = (0.044*t2_12) + (0.844*td2_12) - 0.55

# Extract the number of times within the WRF file and loop over all times in file
      num_times = np.size(extract_times(wrf_in, ALL_TIMES))

# Read in grid and time inforation
      grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']
      sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')
      valid_time = str(extract_times(wrf_in, ALL_TIMES)[i])[0:22]

# Loop over times present in input file
      for i in np.arange(0, num_times, 1):

# Read in cloud fraction and convert to okta value
         cloud_frac = getvar(wrf_in, 'cloudfrac', timeidx=i)[:,min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         cloud_frac_max = np.amax(cloud_frac, axis=0)

         cloud_frac_max_okt = cloud_frac_max * 8.0

# calculate corioilis parameter (f)
         f = 2.0*0.000072921*np.sin((lats*2*np.pi)/360.0)

#  define accereation due to gravity g
         g = np.full_like(f, 9.81)

# Read in height and pressure and interpolate height at 900 hPa
         z = getvar(wrf_in, 'z', timeidx=i, units='m')[:,min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         pressure = getvar(wrf_in, 'pressure', timeidx=i)[:,min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         z_900 = interplevel(z, pressure, 900.0)

# Calculate horizontal gradient of z on 900 hPa level and calculate magnitude of gradient
         dzdn_900 = calc_gradient.calc_gradient(z_900, to_np(lats), to_np(lons))

         dzdn_900_mag = np.sqrt((dzdn_900[0]**2.0) + (dzdn_900[1]**2.0))

# Calculate geostrophic windat 900 m
         Vg = (dzdn_900_mag*g)/f

# Calculate value of A in Craddock and Pritchard radiation fog method
         A = np.zeros_like(Vg, dtype='float')
         A = np.where((Vg <= 12.5) & (cloud_frac_max_okt > 4), A+1.0, A)
         A = np.where((Vg <= 12.5) & (cloud_frac_max_okt > 6), A+0.5, A)

         A = np.where((Vg > 12.5) & (cloud_frac_max_okt <= 2), A-1.5, A)
         A = np.where((Vg > 12.5) & (cloud_frac_max_okt > 4), A+0.5, A)

# Calculate fog point temperature
         Tf = Y+A

# Read in 2m temperature and convert to celsius
         T2 = getvar(wrf_in, 'T2', timeidx=i)[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         T2_C = T2-273.15
   
# Calculate temperature difference between fog point and 2m temperature
         Tdiff = Tf - T2_C

# Create figure and axes
         fig = plt.figure(figsize=(10,10))
         ax = plt.axes(projection=cart_proj)
         ax.coastlines(linewidth=1.0)
         ax.add_feature(feature.OCEAN,facecolor=("lightblue"))
         ax.add_feature(feature.LAND,facecolor=("sandybrown"))
         gl = ax.gridlines(linewidth=0.5, draw_labels=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
         gl.right_labels = False
         gl.bottom_labels = False

# Plot cloud cover

         cloud_lvls = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
         color1 = [1,1,1,0]
         color2 = [1,1,1,1]
#         cloud_cmap = np.linspace(color1, color2, 7)
         cloud_cmap = ListedColormap(np.linspace(color1, color2, 8))
         cloud = plt.contourf(lons, lats, Tdiff, levels=cloud_lvls, cmap=cloud_cmap, zorder=1, antialiased=True, transform=crs.PlateCarree(), extend="max")

         if np.size(lats[:,0]) < np.size(lats[0,:]):
            portrait = True
         else:
            portrait = False

# Create inset colourbar

         cbar_inset = False

         if cbar_inset:

            if portrait:
               cbbox = inset_axes(ax, '10%', '90%', loc = 7)
               [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
               cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
               cbbox.set_facecolor([1,1,1,0.0])
               cbbox.text(0.9,0.5, "Radiation fog (temperature below fog point)", rotation=90.0, verticalalignment='center', horizontalalignment='center')
               cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
               cb = plt.colorbar(cax=cbaxes, aspect=20)
               cb.patch.set_facecolor((.0, 1.0, 1.0, 0.0))
            else:
               cbbox = inset_axes(ax, '90%', '10%', loc = 8)
               [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
               cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
               cbbox.set_facecolor([1,1,1,0.0])
               cbbox.text(0.5,0.1, "Radiation fog (temperature below fog point)", verticalalignment='center', horizontalalignment='center')
               cbaxes = inset_axes(cbbox, '95%', '30%', loc = 9)
               cb = plt.colorbar(cax=cbaxes, orientation='horizontal')
               cb.patch.set_facecolor((.0, 1.0, 1.0, 0.0))
         else:
            cbbox = inset_axes(ax, '100%', '100%', bbox_to_anchor=(0, -0.13, 1, 0.13), bbox_transform=ax.transAxes, loc = 8, borderpad=0)
#            [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
            cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
            cbbox.set_facecolor("lightblue")
            cbbox.text(0.5,0.1, "Radiation fog (temperature below fog point)", verticalalignment='center', horizontalalignment='center')
            cbaxes = inset_axes(cbbox, '95%', '30%', loc = 9)
            cb = plt.colorbar(cax=cbaxes, orientation='horizontal')
            cb.patch.set_facecolor((.0, 1.0, 1.0, 0.0))


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
#   dest_dir = "/home/earajr/FORCE_WRF_plotting/output/radiationfog"
   dest_dir = sys.argv[2]
   if not os.path.isdir(dest_dir):
       os.makedirs(dest_dir)

## Define input directory
#   input_dir = "/home/earajr/FORCE_WRF_plotting/WRF_plot_inputs"
#
   limit_lats = []
   limit_lons = []
   map_names = []
#
#   with open(input_dir+"/map_limit_lats", "r") as file:
#       reader = csv.reader(file)
#       for row in reader:
#           limit_lats.append(row)
#
#   with open(input_dir+"/map_limit_lons", "r") as file:
#       reader = csv.reader(file)
#       for row in reader:
#           limit_lons.append(row)
#
#   with open(input_dir+"/map_names", "r") as file:
#       reader = csv.reader(file)
#       for row in reader:
#           map_names.append(row)
#
#   if (np.shape(limit_lats)[0] == np.shape(limit_lons)[0] == np.size(map_names)):
#      print("Number of map limit latitudes, longitudes and map names is correct continuing with map generation.")
#   else:
#      raise ValueError("The number of map limit latitudes, longitudes or map names in the input directory does not match, please check that the map information provided is correct")

# Input WRF out file as an argument (full path)
   wrf_fil = sys.argv[1]
   base_wrf_fil = os.path.basename(wrf_fil)
   dom = base_wrf_fil.split("_")[1]
   date = base_wrf_fil.split("_")[2]
   time = base_wrf_fil.split("_")[3].replace(":", "-")

# Input map information

   map_names.append(sys.argv[3])
   limit_lats.append(sys.argv[4])
   limit_lats.append(sys.argv[5])
   limit_lons.append(sys.argv[6])
   limit_lons.append(sys.argv[7])

# Create input dictionary for each map and pass it to the map_radiationfog function above

   input_dict = {}
   input_dict["latitudes"] = limit_lats
   input_dict["longitudes"] = limit_lons
   input_dict["infile"] = wrf_fil
   input_dict["locationname"] = map_names

   fig = map_radiationfog(input_dict)

   if fig:

      plt.savefig(dest_dir+"/radiationfog_"+dom+"_"+date+"_"+time+"_"+map_names[0]+".png", bbox_inches='tight')

