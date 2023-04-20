import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
import os
import csv
import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from metpy.units import units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from wrf import (getvar, ll_to_xy, interplevel, interpline, vinterp, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs)

# Define destination directory
dest_dir = "/home/earajr/FORCE_WRF_plotting/output/skewT"
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

# Define input directory

input_dir = "/home/earajr/FORCE_WRF_plotting/WRF_plot_inputs"

# Waypoint lat, lons and altitudes

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

# Check for existance of WRF out file
if not os.path.exists(wrf_fil):
    raise ValueError("Warning! "+wrf_fil+" does not exist.")

# Read WRF out netcdf
wrf_in= Dataset(wrf_fil)

# Extract the number of times within the WRF file and loop over all times in file
num_times = np.size(extract_times(wrf_in, ALL_TIMES))

for i in np.arange(0, num_times, 1):

# Read pressure, temperature, dewpoint, and winds

   hgt = getvar(wrf_in, 'z', timeidx=i, units='m')
   rh = getvar(wrf_in, 'rh', timeidx=i)

   press = getvar(wrf_in, 'pressure', timeidx=i)
   press_sparse = getvar(wrf_in, 'pressure', timeidx=i)[0:39:2,:,:]
   t = getvar(wrf_in, 'tc', timeidx=i)
   td = getvar(wrf_in, "td", timeidx=i)
   u = getvar(wrf_in, "ua", timeidx=i)
   v = getvar(wrf_in, "va", timeidx=i)
   u_sparse = getvar(wrf_in, "ua", timeidx=i)[0:39:2,:,:]
   v_sparse = getvar(wrf_in, "va", timeidx=i)[0:39:2,:,:]

   lats, lons = latlon_coords(press)

# loop through soundings

   for j in np.arange(0, np.shape(sounding_names)[0], 1):

      if not sounding_names[j][0]:
         sounding_names[j] = ["lat"+str(sounding_lats[j][0])+"lon"+str(sounding_lons[j][0])]

      print(sounding_names[j])

      x_y = ll_to_xy(wrf_in, sounding_lats[j], sounding_lons[j])

      if x_y[1] >= 0 and x_y[1] < np.shape(lats)[0] and x_y[0] >= 0 and x_y[0] < np.shape(lats)[1]:
         print("Sounding is in lat lon range of selected simulation")

         hgt_s = hgt[:,x_y[0],x_y[1]]
         rh_s = rh[:,x_y[0],x_y[1]]
         press_s = press[:,x_y[0],x_y[1]] * units.hPa
         t_s = t[:,x_y[0],x_y[1]] * units.degC
         td_s = td[:,x_y[0],x_y[1]] * units.degC
         u_s = u[:,x_y[0],x_y[1]] * units('m/s')
         v_s = v[:,x_y[0],x_y[1]] * units('m/s')

         es = 6.1121*np.exp((18.678-(to_np(t_s)[0:4]/234.5))*(to_np(t_s)[0:4]/(257.14+to_np(t_s)[0:4])))
         ws = (621.97*(es/(to_np(press_s)[0:4]-es)))
         w = np.mean(((to_np(rh_s)[0:4]/100.0)*ws)/1000.0)

         press_s_sparse = (press_sparse[:,x_y[0],x_y[1]] * units.hPa)
         u_s_sparse = (u_sparse[:,x_y[0],x_y[1]] * units('m/s'))
         v_s_sparse = (v_sparse[:,x_y[0],x_y[1]] * units('m/s'))

         fig, ax = plt.subplots(1,1, figsize=(10,10))
         [ax.spines[k].set_visible(False) for k in ax.spines]
         ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
         
         skew = SkewT(fig=fig, rotation=30, aspect=120)
         skew.plot(press_s, t_s, 'r')
         skew.plot(press_s, td_s, 'g')
         skew.plot_barbs(press_s_sparse, u_s_sparse, v_s_sparse)

         prof = to_np(mpcalc.parcel_profile(press_s, t_s[0], np.mean(td_s[0:4])))-273.15
         lcl = float(str(to_np(mpcalc.lcl(press_s[0], t_s[0], np.mean(td_s[0:4])))[0]).split()[0])
         cape, cin = mpcalc.cape_cin(press_s, t_s, td_s, prof*units('degC'))
         lfc = float(str(to_np(mpcalc.lfc(press_s, t_s, td_s))[0]).split()[0])
         equil = float(str(to_np(mpcalc.el(press_s, t_s, td_s, prof*units('degC')))[0]).split()[0])
         lifted = float(str(to_np(mpcalc.lifted_index(press_s, t_s, prof*units('degC')))).split('[')[1].split(']')[0])
         k_index = float(str(to_np(mpcalc.k_index(press_s, t_s, td_s))).split()[0])
         cross_tots = float(str(to_np(mpcalc.cross_totals(press_s, t_s, td_s))).split()[0])
         vertical_tots = float(str(to_np(mpcalc.vertical_totals(press_s, t_s))).split()[0])
         showalter = float(str(to_np(mpcalc.showalter_index(press_s, t_s, td_s))).split('[')[1].split(']')[0])
         tot_tots = cross_tots + vertical_tots

         skew.plot(press_s, prof, 'k', linewidth=2)
         skew.shade_cin(to_np(press_s), to_np(t_s), prof)
         skew.shade_cape(to_np(press_s), to_np(t_s), prof)

         skew.plot_dry_adiabats(linewidth=1)
         skew.plot_moist_adiabats(linewidth=1)
         skew.plot_mixing_lines(mixing_ratio=to_np(w), pressure=[press_s[0], lcl]*units.hPa, linewidth=2, linestyles='solid', colors="black", alpha=1.0)
         skew.ax.set_xlim(-60, 40)
         skew.ax.set_xlabel('Temperature ($^\circ$C)')
         skew.ax.set_ylabel('Pressure (hPa)')

         axin = inset_axes(ax, width='20%', height='20%', bbox_to_anchor=(0.095, 0.0, 1.0, 1.0), bbox_transform=ax.transAxes, loc=3)
         axin.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
         axin.set_title('Wind Profile \n 10m/s increment', fontsize=9, fontweight='bold', horizontalalignment='center')

         h = Hodograph(axin, component_range=30.)
         h.add_grid(increment=10)
         h.plot_colormapped(u_s[0:38], v_s[0:38], hgt_s[0:38])

         ax.set_title('SkewTLogP '+sounding_names[j][0]+"\nLat: "+sounding_lats[j][0].strip()+"   Lon: "+sounding_lons[j][0].strip()+"\nCAPE="+ "{:.0f}".format(float(str(to_np(cape)).split()[0]))+" j/kg, CIN="+ "{:.0f}".format(float(str(to_np(cin)).split()[0]))+" j/kg, LCL={:.0f}".format(lcl)+" hPa, LFC={:.0f}".format(lfc)+" hPa, EQ={:.0f}".format(equil)+" hPa\nLFT IDX={:.0f}".format(lifted)+"$^\circ$C, K IDX={:.0f}".format(k_index)+"$^\circ$C, TOTAL TOTS={:.0f}".format(tot_tots)+"$^\circ$C, SHWTR_IDX={:.0f}".format(showalter)+"$^\circ$C" , horizontalalignment='left', fontsize=10, fontweight='bold', loc='left')


         plt.savefig('test_SkewT_'+sounding_names[j][0]+'.png', bbox_inches='tight')



      else:
         print("Sounding is not in lat lon range of selected simulation")

