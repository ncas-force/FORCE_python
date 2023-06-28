def map_crosssectionlocation(x):
    
   import numpy as np
   from cartopy import crs, feature
   import matplotlib.pyplot as plt
   from netCDF4 import Dataset
   from cartopy.feature import NaturalEarthFeature
#   import os
#   import ninept_smoother
#   import calc_gradient
   from matplotlib.cm import get_cmap
   from matplotlib.colors import ListedColormap
   from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#   from skimage.measure import label
   from pyproj import Geod
#
   from wrf import (getvar, interplevel, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs, ll_to_xy)

   lats = x["latitudes"]
   lons = x["longitudes"]

   crosssection_name = x["locationname"]

# Input WRF out file as an argument (full path)
   wrf_fil = x["infile"]

# Check for existance of WRF out file
   if not os.path.exists(wrf_fil):
      raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
   wrf_in = Dataset(wrf_fil)

# Define geodesy
   geod = Geod(ellps='WGS84')

# Read in grid and time inforation
   grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']
   sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')
   valid_time = str(extract_times(wrf_in, ALL_TIMES)[0])[0:22]

# Read all sea level pressure
   ter = getvar(wrf_in, 'ter', timeidx=0)[:,:]

   ter_lats, ter_lons = latlon_coords(ter)

# Get cartopy projection
   cart_proj = get_cartopy(ter)

# Check that the supplied lat and lon values are within the WRF domain, if not then the plot will not be created.

   x_indices = [to_np(ll_to_xy(wrf_in, lats[i], lons[i]))[0] for i in np.arange(0,len(lats),1) ]
   y_indices = [to_np(ll_to_xy(wrf_in, lats[i], lons[i]))[1] for i in np.arange(0,len(lats),1) ]

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) also identify the required max and min indices 

   points_in_dom = True
   for i in np.arange(0,len(x_indices),1):
      if y_indices[i] >= 0 and y_indices[i] < np.shape(ter)[0] and x_indices[i] >= 0 and x_indices[i] < np.shape(ter)[1]:
         print("Point %.0f" %i +" passed")
      else:
         print("Point %.0f" %i +" did not pass")
         points_in_dom = False
      

   if points_in_dom:
      
# Create figure and axes
      fig = plt.figure(figsize=(10,10))
      ax = plt.axes(projection=cart_proj)
      ax.coastlines(linewidth=1.0)
      ax.add_feature(feature.OCEAN,facecolor=("lightblue"))
      ax.add_feature(feature.LAND,facecolor=("mediumseagreen"))
      gl = ax.gridlines(linewidth=0.5, draw_labels=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
      gl.right_labels = False
      gl.bottom_labels = False

# Map terrain

      whole_cmap = get_cmap('terrain')
      cmap = ListedColormap(whole_cmap(np.linspace(0.2, 1.0, 128)))
      
      ter_lvl = np.arange(50, 1000, 50)
      plt.contourf(ter_lons, ter_lats, ter, levels=ter_lvl, cmap=cmap, zorder=1, transform=crs.PlateCarree(), extend="max")

# Add inset title

      title_inset = False

      if title_inset:

         titlebox = inset_axes(ax, '5%', '90%', loc = 7)
         [titlebox.spines[k].set_visible(False) for k in titlebox.spines]
         titlebox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
         titlebox.set_facecolor([1,1,1,0.7])

         titlebox.text(0.5,0.5, crosssection_name+" position", rotation=90.0, verticalalignment='center', horizontalalignment='center')
      else:
         titlebox = inset_axes(ax, '100%', '100%', bbox_to_anchor=(0, -0.05, 1, 0.05), bbox_transform=ax.transAxes, loc = 8, borderpad=0)
         [titlebox.spines[k].set_visible(False) for k in titlebox.spines]
         titlebox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
         titlebox.set_facecolor([1,1,1,0.7])
         titlebox.text(0.5,0.5, crosssection_name+" position", rotation=0.0, verticalalignment='center', horizontalalignment='center')


      ax.plot([float(lon) for lon in lons], [float(lat) for lat in lats], linewidth=2, color='red',transform=crs.PlateCarree())

# Return figure
      return(fig) 

   else:
      print("Not all points lie inside the chosen domain an image will not be generated")



###############################################################################################################################################################################################
# If the script is called as the main script then this part of the code is exectued first. The plotting section above can be called seperately as a module using a dictionary as the only input.

if __name__ == "__main__":

   import os
   import sys
   import csv
   import numpy as np
   import matplotlib.pyplot as plt

# Define destination directory
#   dest_dir = "/home/earajr/FORCE_WRF_plotting/output/mslp"
   dest_dir = sys.argv[2]
   if not os.path.isdir(dest_dir):
       os.makedirs(dest_dir)

## Define input directory
#   input_dir = "/home/earajr/FORCE_WRF_plotting/WRF_plot_inputs"
#
   lats = []
   lons = []
   crosssection_name = []
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

   crosssection_name.append(sys.argv[3])
   lats.append(sys.argv[4].split(","))
   lons.append(sys.argv[5].split(","))

# Create input dictionary for each map and pass it to the map_mslp function above

   input_dict = {}
   input_dict["latitudes"] = lats[0]
   input_dict["longitudes"] = lons[0]
   input_dict["infile"] = wrf_fil
   input_dict["locationname"] = crosssection_name[0]

   print(input_dict)

   fig = map_crosssectionlocation(input_dict)

   plt.savefig(dest_dir+"/crosssectionlocation_"+dom+"_"+crosssection_name[0]+".png", bbox_inches='tight')

