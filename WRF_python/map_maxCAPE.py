def map_maxCAPE(x):

   import numpy as np
   from cartopy import crs
   import matplotlib.pyplot as plt
   from mpl_toolkits.axes_grid1.inset_locator import inset_axes
   from netCDF4 import Dataset
   import os
   from pyproj import Geod

   from wrf import (getvar, interplevel, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs, ll_to_xy)

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
      x["locationname"] = " "
   map_name = x["locationname"]

# Input WRF out file as an argument (full path)
   wrf_fil = x["infile"]

# Check for existance of WRF out file
   if not os.path.exists(wrf_fil):
      raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
   wrf_in = Dataset(wrf_fil)

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

# Read all relative humidity
      rh_all = getvar(wrf_in, 'rh', timeidx=i)[0,:,:]

# Check that the supplied lat and lon values are within the WRF domain, if not then the plot will not be created.
      if limit_id == "all":
         x1_y1 = (1, 1)
         x2_y2 = (1, 1)
      else:
         x1_y1 = ll_to_xy(wrf_in, limit_lats[0], limit_lons[0])
         x2_y2 = ll_to_xy(wrf_in, limit_lats[1], limit_lons[1])

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) also identify the required max and min indices 
      if x1_y1[1] >= 0 and x1_y1[1] < np.shape(rh_all)[0] and x1_y1[0] >= 0 and x1_y1[0] < np.shape(rh_all)[1] and x2_y2[1] >= 0 and x2_y2[1] < np.shape(rh_all)[0] and x2_y2[0] >= 0 and x2_y2[0] < np.shape(rh_all)[1]:
         cart_proj = get_cartopy(rh_all)
         lats_all, lons_all = latlon_coords(rh_all)

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

# Read CAPE, RH and height 

         cape_2d = getvar(wrf_in, 'cape_2d', timeidx=i)[:,min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         rh = getvar(wrf_in, 'rh', timeidx=i)[:,min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         height = getvar(wrf_in, 'height_agl')[:,min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]

         max_cape = cape_2d[0]
         max_cin = np.where(cape_2d[1] >= 0, cape_2d[1]*-1.0, 0.0)
         lcl = cape_2d[2]
         lfc = cape_2d[3]

         interp_levs = np.linspace(lcl, lfc, num=10)
         rh_interp = []

         for j in np.arange(0,10):
            rh_interp.append(interplevel(rh, height, interp_levs[j]))

         rh_mean = np.nanmean(rh_interp, axis=0)

# Subset latitudes and longitudes
         lats = lats_all[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]
         lons = lons_all[min_lat_ind:max_lat_ind+1, min_lon_ind:max_lon_ind+1]

# Create figure and axes
         fig = plt.figure(figsize=(10,10))
         ax = plt.axes(projection=cart_proj)
         ax.coastlines(linewidth=0.5)
         gl = ax.gridlines(linewidth=0.5, draw_labels=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
         gl.right_labels = False
         gl.bottom_labels = False

# Plot mean RH between LCL and LFC

         rh_lvls = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
         RH = plt.contourf(lons, lats, rh_mean, levels=rh_lvls, colors='none', zorder=2,  hatches=['','','','','','','.','.','.','.'], transform=crs.PlateCarree())

# Plot CIN

         cin_lvls = [-250, -150, -100, -50]
         CIN = plt.contour(lons, lats, max_cin, levels=cin_lvls, colors='red', linestyles='solid', zorder=3, transform=crs.PlateCarree())
         plt.clabel(CIN, CIN.levels, inline=True, fmt= '%1.0f')


# Plot CAPE

         cape_lvls = [60, 80, 100, 120, 150, 175, 200, 240, 290, 340, 400, 500, 600, 700, 850, 1000, 1250, 1500, 1750, 2000, 2500]
         plt.contourf(lons, lats, max_cape, levels=cape_lvls, cmap='magma_r', zorder=1, transform=crs.PlateCarree())

         if np.size(lats[:,0]) < np.size(lats[0,:]):
            portrait = True
         else:
            portrait = False

# Create inset colourbar

         cbar_inset = False

         if cbar_inset:

            if portrait:
               cbbox = inset_axes(ax, '15%', '90%', loc = 7)
               [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
               cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
               cbbox.set_facecolor([1,1,1,0.7])
               cbbox.text(0.82,0.5, "Maximum CAPE (J/kg)", rotation=90.0, verticalalignment='center', horizontalalignment='center')
               cbbox.text(0.95,0.25, "Maximum CIN (J/kg)", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='red')
               cbbox.text(0.95,0.75, u'$\u25CF$'+" RH between LCL and LFC (over 60%)", rotation=90.0, verticalalignment='center', horizontalalignment='center', color='black')
               cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
               cb = plt.colorbar(cax=cbaxes, aspect=20)
            else:
               cbbox = inset_axes(ax, '90%', '12%', loc = 8)
               [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
               cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
               cbbox.set_facecolor([1,1,1,0.7])
               cbbox.text(0.5,0.3, "Maximum CAPE (J/kg)", verticalalignment='center', horizontalalignment='center')
               cbbox.text(0.75,0.15, u'$\u25CF$'+" RH between LCL and LFC (over 60%)", verticalalignment='center', horizontalalignment='center', color='black')
               cbbox.text(0.25,0.15, "Maximum CIN (J/kg)", verticalalignment='center', horizontalalignment='center', color='red')
               cbaxes = inset_axes(cbbox, '95%', '30%', loc = 9)
               cb = plt.colorbar(cax=cbaxes, orientation='horizontal')
         else:
            cbbox = inset_axes(ax, '100%', '100%', bbox_to_anchor=(0, -0.13, 1, 0.13), bbox_transform=ax.transAxes, loc = 8, borderpad=0)
            [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
            cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
            cbbox.set_facecolor([1,1,1,0.7])
            cbbox.text(0.5,0.3, "Maximum CAPE (J/kg)", verticalalignment='center', horizontalalignment='center')
            cbbox.text(0.75,0.15, u'$\u25CF$'+" RH between LCL and LFC (over 60%)", verticalalignment='center', horizontalalignment='center', color='black')
            cbbox.text(0.25,0.15, "Maximum CIN (J/kg)", verticalalignment='center', horizontalalignment='center', color='red')
            cbaxes = inset_axes(cbbox, '95%', '30%', loc = 9)
            cb = plt.colorbar(cax=cbaxes, orientation='horizontal')

# Add inset timestamp
         tsbox = inset_axes(ax, '95%', '3%', loc = 9)
         [tsbox.spines[k].set_visible(False) for k in tsbox.spines]
         tsbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
         tsbox.set_facecolor([1,1,1,1])

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
   dest_dir = "/home/earajr/FORCE_WRF_plotting/output/maxCAPE"
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
      print("Number of map limit latitudes, longitudes and map names is correct continuing with cross section generation.")
   else:
      raise ValueError("The number of map limit latitudes, longitudes or map names in the input directory does not match, please check that the map information provided is correct")

# Input WRF out file as an argument (full path)
   wrf_fil = sys.argv[1]

# Loop through maps, create input dictionary for each map and pass it to the map_maxCAPE function above
   for i in np.arange(0, np.shape(limit_lats)[0], 1):
      input_dict = {}
      input_dict["latitudes"] = limit_lats[i]
      input_dict["longitudes"] = limit_lons[i]
      input_dict["infile"] = wrf_fil
      input_dict["locationname"] = map_names[i]

      fig = map_maxCAPE(input_dict)

      plt.savefig(dest_dir+"/maxCAPEtest_"+map_names[i][0]+".png", bbox_inches='tight')
