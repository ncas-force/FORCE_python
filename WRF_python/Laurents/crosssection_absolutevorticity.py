def crosssection_absolutevorticity(x):

   import numpy as np
   import matplotlib.pyplot as plt
   from netCDF4 import Dataset
   import os
   from pyproj import Geod
   import fill_xsec_nans
   import rotate_ua_va_vert_cross

   from wrf import (getvar, interplevel, interpline, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs)

# Contour information

   cmin = -155.0
   cmax = 155.0
   cstep = 5.0

# Read cross section spatial information from input dictionary 

   xsection_lats = x["latitudes"]
   xsection_lats_str = ['{:.2f}'.format(x) for x in xsection_lats]
   xsection_lons = x["longitudes"]
   xsection_lons_str = ['{:.2f}'.format(x) for x in xsection_lons]
   if "alts" not in x:
      x["alts"] = [0]*len(x["latitudes"])
   xsection_alts = x["alts"]
   base_alt = x["levels"][0]
   top_alt = x["levels"][1]
   if "locationname" not in x:
      x["locationname"] = " "
   crosssectionname = x["locationname"]

# Input WRF out file as an argument (full path)
   wrf_fil = x["infile"]

# Read wind vector flag to chackj if wind vectors are required.
   wind_flag = x["windvector"]

# Check for existance of WRF out file
   if not os.path.exists(wrf_fil):
       raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
   wrf_in= Dataset(wrf_fil)

# Extract the number of times within the WRF file and loop over all times in file
   num_times = np.size(extract_times(wrf_in, ALL_TIMES))

# Calculate levels to interpolate onto
   lev_interval = (float(top_alt) - float(base_alt))/100.0
   interp_levs = np.arange(float(base_alt), float(top_alt)+lev_interval, lev_interval)

# Check that the number of waypoints in the cross section for lat lon and altitude match each other. Only proceed if they do.
   if (np.size(xsection_lats) == np.size(xsection_lons) == np.size(xsection_alts)):

# Calculate great circle distance between waypoints and create cross section locations based on waypoint lats and lons
      geod = Geod(ellps='WGS84')

      xsection_locs = []
      gc_dist = []

      for j in np.arange(0, np.size(xsection_lats), 1):
         xsection_locs.append(CoordPair(lat=xsection_lats[j], lon=xsection_lons[j]))
         if j > 0:
            gc_dist.append(geod.inv(xsection_lons[j-1], xsection_lats[j-1], xsection_lons[j], xsection_lats[j])[2]/1000.0)

# Calculate total length of cross section and percentage of each section (including integer version for plot scaling)
      total_dist = np.sum(gc_dist)
      dist_percent = 100.0*(gc_dist/total_dist)
      dist_percent_int = []
      for j in np.arange(0, np.size(dist_percent), 1):
         dist_percent_int.append(int(dist_percent[j]))
      dist_percent_int.append(1)
      dist_percent_int.append(3)

# Loop over times present in input file
      for j in np.arange(0, num_times, 1):

# Read grid and times information
         grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']
         sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')
         valid_time = str(extract_times(wrf_in, ALL_TIMES)[j])[0:22]

# Read height, temperature and terrain
         ht = getvar(wrf_in, 'z', timeidx=j)
         avo = getvar(wrf_in, 'avo', timeidx=j)
         ter = getvar(wrf_in, "ter", timeidx=-1)

         if wind_flag:
            ua = getvar(wrf_in, 'ua', timeidx=j)
            va = getvar(wrf_in, 'va', timeidx=j)
            wa = getvar(wrf_in, 'wa', timeidx=j)

# Compute vertical cross section interpolation
         avo_cross = []
         ter_cross = []

         if wind_flag:
            h_wind_cross = []
            wa_cross = []

         for k in np.arange(0, np.size(gc_dist), 1):
            avo_cross.append(vertcross(avo, ht, wrfin=wrf_in, levels=interp_levs, start_point=xsection_locs[k], end_point=xsection_locs[k+1], latlon=True, meta=True))
            ter_cross.append(interpline(ter, wrfin=wrf_in, start_point=xsection_locs[k], end_point=xsection_locs[k+1]))

            if wind_flag:

               ua_cross = vertcross(ua, ht, wrfin=wrf_in, levels=interp_levs, start_point=xsection_locs[k], end_point=xsection_locs[k+1], latlon=True, meta=True)
               va_cross = vertcross(va, ht, wrfin=wrf_in, levels=interp_levs, start_point=xsection_locs[k], end_point=xsection_locs[k+1], latlon=True, meta=True)
               wa_cross.append(vertcross(wa, ht, wrfin=wrf_in, levels=interp_levs, start_point=xsection_locs[k], end_point=xsection_locs[k+1], latlon=True, meta=True))

               h_wind_cross.append(rotate_ua_va_vert_cross.rotate(ua_cross, va_cross)[0])

# Read projection from a variable (will be able to detect all possible WRF projections and use them for plotting) 
         cart_proj = get_cartopy(avo)
         lats, lons = latlon_coords(avo)

# Create figure and axes remove wspace and hspace to make panels butt against each other.
         fig = plt.figure(figsize=(15,5))
         plt.subplots_adjust(wspace=0, hspace=0)

# create gridspec for subplot scaling
         gs = fig.add_gridspec(1,np.size(gc_dist)+2, width_ratios=dist_percent_int)

# Set contour levels
         avo_lvls = np.arange(cmin, cmax, cstep)
         avo_lvls2 = np.arange(cmin, cmax, 2.0*cstep)

# loop through parts of cross section and plot data
         for k in np.arange(0, np.size(gc_dist), 1):
            ax = fig.add_subplot(gs[k])
            ax.axis(ymin=0.0, ymax=100.0)
            vert_vals = to_np(avo_cross[k].coords["vertical"])
            v_ticks = np.arange(vert_vals.shape[0])
            thin = [int(x/10.0) for x in v_ticks.shape]
            ax.set_yticks(v_ticks[::thin[0]])
            x_ticks = np.arange(0.0, np.shape(avo_cross[k])[1], 1)

            if k == 0:
               if wind_flag and np.size(gc_dist) <= 1:
                  plt_title = 'Absolute Vorticity cross section and wind vectors parallel to cross section:'
               else:
                  plt_title = 'Absolute Vorticity cross section:'
               plt.title(plt_title+' '+crosssectionname+'\nLatitudes (degrees north): '+','.join(xsection_lats_str)+ ', Longitudes (degrees east): '+','.join(xsection_lons_str)+'\nSimulation start time: '+sim_start_time['SIMULATION_START_DATE'].replace("_", " ")[0:19]+', Valid time: '+str(valid_time).replace("T", " ")[0:19], horizontalalignment='left', fontsize=10, fontweight='bold', loc='left')
            
            avo_cross_np = np.where(np.isnan(to_np(avo_cross[k])), np.nan ,to_np(avo_cross[k]))

            if wind_flag:
               h_wind_cross_np = np.where(np.isnan(to_np(h_wind_cross[k])), np.nan, to_np(h_wind_cross[k]))
               wa_cross_np = np.where(np.isnan(to_np(wa_cross[k])), np.nan, to_np(wa_cross[k]))

# Fill in nans at the bottom of the cross section with nearest value from above
            avo_cross_np = fill_xsec_nans.fill_xsec_nans(avo_cross_np)

# Create contours 
            t_contours = ax.contourf(x_ticks, v_ticks, avo_cross_np, levels=avo_lvls, cmap='bwr')
            t_contours2 = ax.contour(x_ticks, v_ticks, avo_cross_np, levels=avo_lvls2, colors='k')

            if wind_flag:

               if np.size(gc_dist) <= 1:
                  thin_vec = [int(x/20.) for x in h_wind_cross_np.shape]
                  ws_cross_np = np.sqrt((h_wind_cross_np**2.0)+(wa_cross_np**2.0))
                  max_ws_cross_np = np.nanmax(ws_cross_np)

                  if max_ws_cross_np <=12.5:
                     key_val = 10.0
                  elif max_ws_cross_np <= 30.0:
                     key_val = 25.0
                  elif max_ws_cross_np <= 52.0:
                     key_val = 50.0
                  elif max_ws_cross_np <= 73.0:
                     key_val = 75.0
                  elif max_ws_cross_np <= 95.0:
                     key_val = 100.0
                  elif max_ws_cross_np <= 113.0:
                     key_val = 125.0
                  else:
                     key_val = 150.0
            
                  scale = key_val/8.0

                  h_wind_quivers = ax.quiver(x_ticks[::thin_vec[1]], v_ticks[::thin_vec[0]], h_wind_cross_np[::thin_vec[0],::thin_vec[1]], wa_cross_np[::thin_vec[0],::thin_vec[1]], pivot='middle', zorder=2, width=0.002, scale_units='y', scale=scale)
                  quiverkey = ax.quiverkey(h_wind_quivers, 0.9, 0.9, key_val, '{:.0f} '.format(key_val)+'m/s', labelpos='E', coordinates='figure')

# correct waypoint altitudes so that they cannot be below the suface (unless all are set to 0)
            if not all(float(v) == 0.0 for v in xsection_alts):

               if float(xsection_alts[k]) < to_np(ter_cross[k])[0]:
                  xsection_alts1 = to_np(ter_cross[k])[0]
               else:
                  xsection_alts1 = float(xsection_alts[k])

               if float(xsection_alts[k+1]) < to_np(ter_cross[k])[-1]:
                  xsection_alts2 = to_np(ter_cross[k])[-1]
               else:
                  xsection_alts2 = float(xsection_alts[k+1])

# Plot polyline indicating path of aircraft through the cross section
               ax.plot([0, x_ticks[-1]], [100.0*((xsection_alts1-float(base_alt))/(float(top_alt)-float(base_alt))), 100.0*((xsection_alts2-float(base_alt))/(float(top_alt)-float(base_alt)))], color='red')

# Plot terrain at the bottom of the cross section
            ht_fill = ax.fill_between(x_ticks, 0, 100.0*((to_np(ter_cross[k])-float(base_alt))/(float(top_alt)-float(base_alt))), facecolor="black")

# Setup tickmarks so that only the first panel has y axis labels and x axis labels are lat and lon values

            step = 10
            num_x_ticks = dist_percent[k]/step
            x_tick_step = round(np.shape(avo_cross_np)[1]/num_x_ticks)
            indices = np.arange(0, np.shape(avo_cross[k])[1], x_tick_step)
            while True:
               if np.shape(avo_cross_np)[1] - indices[-1] < x_tick_step*0.75:
                  step += 1
                  num_x_ticks = dist_percent[k]/step
                  x_tick_step = round(np.shape(avo_cross_np)[1]/num_x_ticks)
                  indices = np.arange(0, np.shape(avo_cross_np)[1], x_tick_step)
               else:
                  break

            if k == 0 :
               ax.set_yticklabels(vert_vals[::thin[0]], fontsize=10)
               ax.set_xticks(x_ticks[0::x_tick_step])
               coords = to_np(avo_cross[k].coords["xy_loc"])
               lat_labels = [(x.lat) for x in coords]
               lat_labels_str = ['{:.2f}'.format(x) for x in lat_labels]
               lon_labels = [(x.lon) for x in coords]
               lon_labels_str = ['{:.2f}'.format(x) for x in lon_labels]
               labels = [str(lat_labels_str[x])+"\n"+str(lon_labels_str[x]) for x in np.arange(0, np.shape(avo_cross[k])[1], x_tick_step)]
               if k == np.size(gc_dist)-1:
                  x_ticks_temp = list(x_ticks[0::x_tick_step])
                  x_ticks_temp.append(x_ticks[-1])
                  ax.set_xticks(x_ticks_temp)
                  labels.append(str(lat_labels_str[-1])+"\n"+str(lon_labels_str[-1]))
               ax.set_xticklabels(labels, fontsize=10)
            elif k != np.size(gc_dist)-1:
               ax.set_xticks(x_ticks[0::x_tick_step])
               coords = to_np(avo_cross[k].coords["xy_loc"])
               lat_labels = [(x.lat) for x in coords]
               lat_labels_str = ['{:.2f}'.format(x) for x in lat_labels]
               lon_labels = [(x.lon) for x in coords]
               lon_labels_str = ['{:.2f}'.format(x) for x in lon_labels]
               labels = [str(lat_labels_str[x])+"\n"+str(lon_labels_str[x]) for x in np.arange(0, np.shape(avo_cross[k])[1], x_tick_step)]
               ax.set_xticklabels(labels, fontsize=10)
            else:
               ax.set_yticklabels([])
               x_ticks_temp = list(x_ticks[0::x_tick_step])
               x_ticks_temp.append(x_ticks[-1])
               ax.set_xticks(x_ticks_temp)
               coords = to_np(avo_cross[k].coords["xy_loc"])
               lat_labels = [(x.lat) for x in coords]
               lat_labels_str = ['{:.2f}'.format(x) for x in lat_labels]
               lon_labels = [(x.lon) for x in coords]
               lon_labels_str = ['{:.2f}'.format(x) for x in lon_labels]
               labels = [str(lat_labels_str[x])+"\n"+str(lon_labels_str[x]) for x in np.arange(0, np.shape(avo_cross[k])[1], x_tick_step)]
               labels.append(str(lat_labels_str[-1])+"\n"+str(lon_labels_str[-1]))
               ax.set_xticklabels(labels, fontsize=10)

            ax.grid(axis = 'y')

            fig.text(0.075, 0.5, "Altitude (meters)", va='center', rotation='vertical')
            fig.text(0.42, 0.00, "Latitude (degrees N) \n      Longitude (degrees E)", va='center')

# Set position of contour labels so that they only occur in panels that are greater than 15 % of the total width and occur down tha middle of the panel
            if any(span >= 15 for span in dist_percent_int):
               if dist_percent_int[k] >= 15:
                  mid_x = float(x_ticks[-1])/2.0
                  clab_locs = []
                  for line in t_contours2.collections:
                     for path in line.get_paths():
                        verts = path.vertices
                 
                        x_val = verts[np.abs(mid_x-verts[:,0]).argmin(),0]
                        if x_val == 0 or x_val == x_ticks[-1]:
                           x_val = mid_x
                 
                        clab_locs.append([x_val,np.mean(verts[:,1])])

                  ax.clabel(t_contours2, fontsize=9, inline=1, fmt='%2.1f', manual=clab_locs)
            else:
               thresh = sorted(dist_percent)[-2]
               if dist_percent[k] >= thresh:
                  mid_x = float(x_ticks[-1])/2.0
                  clab_locs = []
                  for line in t_contours2.collections:
                     for path in line.get_paths():
                        verts = path.vertices

                        x_val = verts[np.abs(mid_x-verts[:,0]).argmin(),0]
                        if x_val == 0 or x_val == x_ticks[-1]:
                           x_val = mid_x

                        clab_locs.append([x_val,np.mean(verts[:,1])])

                  ax.clabel(t_contours2, fontsize=9, inline=1, fmt='%2.1f', manual=clab_locs)

# Add additional panel to act as a colorbar
         ax = fig.add_subplot(gs[k+1])
         ax.set_yticklabels([])
         ax.set_xticklabels([])
         ax.axis('off')
         ax = fig.add_subplot(gs[k+2])
         fig.colorbar(t_contours, cax=ax, label="Absolue Vorticity $10^{-5}s^{-1}$")
         
# Return figure
         return(fig)
   else:
      print("The lat, lon and altitude waypoint arrays supplied were of different sizes, this cross section cannot be generated.")





###############################################################################################################################################################################################
# If the script is called as the main script then this part of the code is exectued first. The plotting section above can be called seperately as a module using a dictionary as the only input. 

if __name__ == "__main__":

   import os
   import sys
   import csv
   import numpy as np
   import matplotlib.pyplot as plt

# Define destination directory
   dest_dir = "/home/earajr/FORCE_WRF_plotting/output/absvort"
   if not os.path.isdir(dest_dir):
       os.makedirs(dest_dir)

# Define input directory
   input_dir = "/home/earajr/FORCE_WRF_plotting/WRF_plot_inputs"

# Waypoint lat, lons and altitudes
   wp_lats = []
   wp_lons = []
   wp_alts = []
   base_alts = []
   top_alts = []
   xsection_ids = []

   with open(input_dir+"/xsection_lats", "r") as file:
       reader = csv.reader(file)
       for row in reader:
           wp_lats.append(row)

   with open(input_dir+"/xsection_lons", "r") as file:
       reader = csv.reader(file)
       for row in reader:
           wp_lons.append(row)

   with open(input_dir+"/xsection_alts", "r") as file:
       reader = csv.reader(file)
       for row in reader:
           wp_alts.append(row)

   with open(input_dir+"/base_alts", "r") as file:
       reader = csv.reader(file)
       for row in reader:
           base_alts.append(row)

   with open(input_dir+"/top_alts", "r") as file:
       reader = csv.reader(file)
       for row in reader:
           top_alts.append(row)

   with open(input_dir+"/xsection_ids", "r") as file:
       reader = csv.reader(file)
       for row in reader:
           xsection_ids.append(row)

# Check to see if the number of arrays provided in lat, lon and alt files match
   if (np.shape(wp_lats)[0] == np.shape(wp_lons)[0] == np.shape(wp_alts)[0] == np.size(base_alts) == np.size(top_alts)== np.size(xsection_ids)):
      print("Number of cross section arrays and base and top altitudes provided is correct continuing with cross section generation.")
   else:
      raise ValueError("The number of cross section arrays or base/top altitudes in the input directory does not match, please check that the cross section information provided is correct")

# Input WRF out file as an argument (full path)
   wrf_fil = sys.argv[1]

# Loop through cross sections, create input dictionary for each cross section and pass it to the crossection_absolutevorticity function above
   for i in np.arange(0, np.shape(wp_lats)[0], 1):
      input_dict = {}
      input_dict["latitudes"] = wp_lats[i]
      input_dict["longitudes"] = wp_lons[i]
      input_dict["leveltype"] = "altitude"
      input_dict["levels"] = [base_alts[i][0], top_alts[i][0]]
      input_dict["alts"] = wp_alts[i]
      input_dict["infile"] = wrf_fil
      input_dict["windvector"] = "True"
      input_dict["locationname"] = xsection_ids[i][0]

      fig = crosssection_absolutevorticity(input_dict)

      plt.savefig(dest_dir+"/test_"+str(xsection_ids[i][0])+".png", bbox_inches='tight')


