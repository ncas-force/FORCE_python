def profile_skewT(x):

   import numpy as np
   import matplotlib.pyplot as plt
   from netCDF4 import Dataset
   import os
#  from pyproj import Geod
   import metpy.calc as mpcalc
   from metpy.plots import SkewT, Hodograph
   from metpy.units import units
   from mpl_toolkits.axes_grid1.inset_locator import inset_axes

   from wrf import (getvar, ll_to_xy, interplevel, interpline, vinterp, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs)

   profile_lat = x["latitudes"]
   profile_lon = x["longitudes"]
   if "locationname" not in x:
      x["locationname"] = " " 
   profile_name = x["locationname"]

# Input WRF out file as an argument (full path)
   wrf_fil = x["infile"]

# Check for existance of WRF out file
   if not os.path.exists(wrf_fil):
      raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
   wrf_in= Dataset(wrf_fil)

# Extract the number of times within the WRF file and loop over all times in file
   num_times = np.size(extract_times(wrf_in, ALL_TIMES))

# Loop over times present in input file
   for i in np.arange(0, num_times, 1):

# Read in grid and time inforation
      grid_id = extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']
      sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')
      valid_time = str(extract_times(wrf_in, ALL_TIMES)[i])[0:22]

# Read pressure, temperature, dewpoint, and winds
      hgt = getvar(wrf_in, 'z', timeidx=i, units='m')
      rh = getvar(wrf_in, 'rh', timeidx=i)

      press = getvar(wrf_in, 'pressure', timeidx=i)
      t = getvar(wrf_in, 'tc', timeidx=i)
      td = getvar(wrf_in, "td", timeidx=i)
      u = getvar(wrf_in, "ua", timeidx=i)
      v = getvar(wrf_in, "va", timeidx=i)

# Read in sparse u, v and pressure to reduce wind levels
      press_sparse = getvar(wrf_in, 'pressure', timeidx=i)[0:39:2,:,:]
      u_sparse = getvar(wrf_in, "ua", timeidx=i)[0:39:2,:,:]
      v_sparse = getvar(wrf_in, "va", timeidx=i)[0:39:2,:,:]

      lats, lons = latlon_coords(press)

# Substitute sounding name with lat and lon if empty, this is now redundant as the lat and lon of the profile will always be included.
#      if not sounding_names[j][0]:
#         sounding_names[j] = ["lat"+str(sounding_lats[j][0])+"lon"+str(sounding_lons[j][0])]

      x_y = ll_to_xy(wrf_in, float(profile_lat[0]), float(profile_lon[0]))

      if x_y[1] >= 0 and x_y[1] < np.shape(lats)[0] and x_y[0] >= 0 and x_y[0] < np.shape(lats)[1]:
         print("Sounding is in lat lon range of selected simulation")

# Subset data for specific profile 
         hgt_p = hgt[:,x_y[0],x_y[1]]
         rh_p = rh[:,x_y[0],x_y[1]]
         press_p = press[:,x_y[0],x_y[1]] * units.hPa
         t_p = t[:,x_y[0],x_y[1]] * units.degC
         td_p = td[:,x_y[0],x_y[1]] * units.degC
         u_p = u[:,x_y[0],x_y[1]] * units('m/s')
         v_p = v[:,x_y[0],x_y[1]] * units('m/s')

         es_p = 6.1121*np.exp((18.678-(to_np(t_p)[0:4]/234.5))*(to_np(t_p)[0:4]/(257.14+to_np(t_p)[0:4])))
         ws_p = (621.97*(es_p/(to_np(press_p)[0:4]-es_p)))
         w_p = np.mean(((to_np(rh_p)[0:4]/100.0)*ws_p)/1000.0)

         press_p_sparse = (press_sparse[:,x_y[0],x_y[1]] * units.hPa)
         u_p_sparse = (u_sparse[:,x_y[0],x_y[1]] * units('m/s'))
         v_p_sparse = (v_sparse[:,x_y[0],x_y[1]] * units('m/s'))

# Create figure and axes 
         fig, ax = plt.subplots(1,1, figsize=(10,10))
         [ax.spines[k].set_visible(False) for k in ax.spines]
         ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)

         skew = SkewT(fig=fig, rotation=30, aspect=120)
         skew.plot(press_p, t_p, 'r')
         skew.plot(press_p, td_p, 'g')
         skew.plot_barbs(press_p_sparse, u_p_sparse, v_p_sparse)

# Calculate profiles and stability metrics
         prof = to_np(mpcalc.parcel_profile(press_p, t_p[0], np.mean(td_p[0:4])))-273.15
         lcl = float(str(to_np(mpcalc.lcl(press_p[0], t_p[0], np.mean(td_p[0:4])))[0]).split()[0])
         cape, cin = mpcalc.cape_cin(press_p, t_p, td_p, prof*units('degC'))
         lfc = float(str(to_np(mpcalc.lfc(press_p, t_p, td_p))[0]).split()[0])
         equil = float(str(to_np(mpcalc.el(press_p, t_p, td_p, prof*units('degC')))[0]).split()[0])
         lifted = float(str(to_np(mpcalc.lifted_index(press_p, t_p, prof*units('degC')))).split('[')[1].split(']')[0])
         k_index = float(str(to_np(mpcalc.k_index(press_p, t_p, td_p))).split()[0])
         cross_tots = float(str(to_np(mpcalc.cross_totals(press_p, t_p, td_p))).split()[0])
         vertical_tots = float(str(to_np(mpcalc.vertical_totals(press_p, t_p))).split()[0])
         showalter = float(str(to_np(mpcalc.showalter_index(press_p, t_p, td_p))).split('[')[1].split(']')[0])
         tot_tots = cross_tots + vertical_tots

# Plot skewT
         skew.plot(press_p, prof, 'k', linewidth=2)
         skew.shade_cin(to_np(press_p), to_np(t_p), prof)
         skew.shade_cape(to_np(press_p), to_np(t_p), prof)

         skew.plot_dry_adiabats(linewidth=1)
         skew.plot_moist_adiabats(linewidth=1)
         skew.plot_mixing_lines(mixing_ratio=to_np(w_p), pressure=[press_p[0], lcl]*units.hPa, linewidth=2, linestyles='solid', colors="black", alpha=1.0)
         skew.ax.set_xlim(-60, 40)
         skew.ax.set_xlabel('Temperature ($^\circ$C)')
         skew.ax.set_ylabel('Pressure (hPa)')

# Create inset axes and add hodograph
         axin = inset_axes(ax, width='20%', height='20%', bbox_to_anchor=(0.095, 0.0, 1.0, 1.0), bbox_transform=ax.transAxes, loc=3)
         axin.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
         axin.set_title('Wind Profile \n 10m/s increment', fontsize=9, fontweight='bold', horizontalalignment='center')

         h = Hodograph(axin, component_range=30.)
         h.add_grid(increment=10)
         h.plot_colormapped(u_p[0:38], v_p[0:38], hgt_p[0:38])

# Add title
         ax.set_title('SkewTLogP '+profile_name+'\nLat: '+'{:.2f}'.format(float(str(profile_lat[0])))+'   Lon: ''{:.2f}'.format(float(str(profile_lon[0])))+'\nSimulation start time: '+sim_start_time['SIMULATION_START_DATE']+', Valid time: '+str(valid_time)+'\nCAPE='+ '{:.0f}'.format(float(str(to_np(cape)).split()[0]))+' j/kg, CIN='+ '{:.0f}'.format(float(str(to_np(cin)).split()[0]))+' j/kg, LCL={:.0f}'.format(lcl)+' hPa, LFC={:.0f}'.format(lfc)+' hPa, EQ={:.0f}'.format(equil)+' hPa\nLFT IDX={:.0f}'.format(lifted)+'$^\circ$C, K IDX={:.0f}'.format(k_index)+'$^\circ$C, TOTAL TOTS={:.0f}'.format(tot_tots)+'$^\circ$C, SHWTR_IDX={:.0f}'.format(showalter)+'$^\circ$C' , horizontalalignment='left', fontsize=10, fontweight='bold', loc='left')


# Return figure
         return(fig)

      else:
         print("Sounding is not in lat lon range of selected simulation")

##############################################################################################################################################################################################
# If the script is called as the main script then this part of the code is exectued first. The plotting section above can be called seperately as a module using a dictionary as the only input. 

if __name__ == "__main__":

   import os
   import sys
   import csv
   import numpy as np
   import matplotlib.pyplot as plt

# Define destination directory
   dest_dir = "/home/earajr/FORCE_WRF_plotting/output/skewT"
   if not os.path.isdir(dest_dir):
       os.makedirs(dest_dir)

# Define input directory
   input_dir = "/home/earajr/FORCE_WRF_plotting/WRF_plot_inputs"

# Profile location

   sounding_lats = []
   sounding_lons = []
   sounding_names = []

   with open(input_dir+"/sounding_lats", "r") as file:
      reader = csv.reader(file)
      for row in reader:
         sounding_lats.append(row)

   with open(input_dir+"/sounding_lons", "r") as file:
      reader = csv.reader(file)
      for row in reader:
         sounding_lons.append(row)

   with open(input_dir+"/sounding_names", "r") as file:
      reader = csv.reader(file)
      for row in reader:
         sounding_names.append(row)

# Check to see if the number of arrays provided in lat, lon and alt files match
   if (np.shape(sounding_lats) == np.shape(sounding_lons) == np.shape(sounding_names)):
      print("Number sounding latitudes, longitudes and names is correct continuing with skewT generation.")
   else:
      raise ValueError("The number of sounding latitudes, longitudes and names does not match, please check that the sounding information provided is correct")

# Input WRF out file as an argument (full path)
   wrf_fil = sys.argv[1]

# Loop through soundings and create dictionary for each cross section and pass it to the profile_skewT function above

   for i in np.arange(0, np.shape(sounding_lats)[0], 1):
      input_dict = {}
      input_dict["latitudes"] = sounding_lats[i]
      input_dict["longitudes"] = sounding_lons[i]
      input_dict["locationname"] = sounding_names[i][0]
      input_dict["infile"] = wrf_fil

      fig = profile_skewT(input_dict)

      plt.savefig(dest_dir+"/test_"+str(sounding_names[i][0])+".png", bbox_inches='tight')

