def timeline_meteogram(x):

   import numpy as np
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   from netCDF4 import Dataset
   import os
   from pyproj import Geod
   import fill_xsec_nans
   import rotate_ua_va_vert_cross
   from datetime import date, datetime, timedelta
   import glob
   from astral import LocationInfo
   from astral.sun import sun
   import pytz
   import pandas as pd
   import matplotlib.dates as mdates
   from matplotlib.patches import Rectangle
   from matplotlib.dates import DateFormatter, HourLocator
   import matplotlib.ticker as mticker
   
   from wrf import (getvar, ll_to_xy, interplevel, interpline, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs)

   def get_pressure_axis_settings(mslp_data, step=4, pad=2):
      """
      Calculate MSLP axis limits and ticks in 4 hPa steps.

      Args:
          mslp_data (array-like): List or array of MSLP values in hPa.
          step (int): Tick interval (default 4 hPa).
          pad (int): Padding in hPa to add before rounding.

      Returns:
          (ymin, ymax, locator): Rounded limits and ticker locator.
      """

      min_val = np.min(mslp_data) - pad
      max_val = np.max(mslp_data) + pad

      # Round limits to nearest multiple of step
      ymin = step * (min_val // step)
      ymax = step * ((max_val + step - 1) // step)

      # Create a ticker with 4 hPa intervals
      locator = mticker.MultipleLocator(base=step)

      return int(ymin), int(ymax), locator


# Read in filename and timeline length and use this information to find simulation start datetime, the domain and the file datetime.

   wrf_fil = x["infile"] # This is the starting file to be read (other subsequent files also need to be read in)
   wrf_in = Dataset(wrf_fil)
   sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')
   timeline_starttime = x["timelinestarttime"]
   timeline_length = x["timelinelength"] 
   base_wrf_fil = os.path.basename(wrf_fil)
   dom = base_wrf_fil.split("_")[1]

   timeline_lat = x["latitudes"]
   timeline_lon = x["longitudes"]

   location_name = x["locationname"]
   

   location = LocationInfo(name="Custom", region="Nowhere", timezone="UTC", latitude=timeline_lat[0], longitude=timeline_lon[0])
   s = sun(observer=location.observer, date=date(int(timeline_starttime[0:4]), int(timeline_starttime[5:7]), int(timeline_starttime[8:10])), tzinfo=pytz.timezone("UTC"))

   sunrise = s['sunrise']
   sunset = s['sunset']

   sim_start_datetime = sim_start_time['SIMULATION_START_DATE']
   timeline_endtime = (datetime(int(timeline_starttime[0:4]), int(timeline_starttime[5:7]), int(timeline_starttime[8:10]), int(timeline_starttime[11:13]), int(timeline_starttime[14:16])) + timedelta(hours = int(timeline_length))).strftime('%Y-%m-%d_%H:%M:%S')

   wrf_path = os.path.split(wrf_fil)[0]
   infils_all = sorted(glob.glob(wrf_path+"/wrfout_"+dom+"*"))
   infils = []

   # Variables for meteogram 2m temperature, precip, mslp, cloud cover (fog, low, mid, high fraction), windspeed (mph), wind direction, weather icons, 1 every 3 hours, cloud base, CAPE, wavestate, sunrise & sunset

   times = []
   T2 = []
   precip_rate = []
   accum_precip = []
   mslp = []
   low_cloud_fraction = []
   mid_cloud_fraction = []
   high_cloud_fraction = []
   cloud_base = []
   windspeed = []
   winddirection = []

   for f in infils_all:
        base = os.path.basename(f)
        try:
            file_time = base.split('_')[2]+"_"+base.split('_')[3]
            if datetime.strptime(timeline_starttime, "%Y-%m-%d_%H:%M:%S") <= datetime.strptime(file_time, "%Y-%m-%d_%H:%M:%S") <= datetime.strptime(timeline_endtime, "%Y-%m-%d_%H:%M:%S"):
                infils.append(f)
        except (IndexError, ValueError):
            # Skip if the filename does not match expected pattern
            continue

   count = 0
   for i, infil in enumerate(infils):
      baseinfil = os.path.basename(infil)
      wrf_in=Dataset(infil)

      if i == 0:
         T2_temp = getvar(wrf_in, 'T2', timeidx=0) # Read in 2m Temperature
         cart_proj = get_cartopy(T2_temp)
         lats, lons = latlon_coords(T2_temp)

         x_y = ll_to_xy(wrf_in, float(timeline_lat[0]), float(timeline_lon[0]))

      all_times = extract_times(wrf_in, timeidx=ALL_TIMES)

      for t_idx, wrf_time in enumerate(all_times):
          current_time = str(wrf_time)[0:19]

          if datetime.strptime(timeline_starttime, "%Y-%m-%d_%H:%M:%S") <= datetime.strptime(current_time, "%Y-%m-%dT%H:%M:%S") <= datetime.strptime(timeline_endtime, "%Y-%m-%d_%H:%M:%S"):

             # Append current time to times list
             times.append(current_time)

             # Calculate precipitation rate and append to precip_rate list
             rainc = getvar(wrf_in, 'RAINC', timeidx=t_idx)[x_y[1], x_y[0]]
             rainnc = getvar(wrf_in, 'RAINNC', timeidx=t_idx)[x_y[1], x_y[0]]
             total_accum_rain = float(rainc + rainnc)

             accum_precip.append(total_accum_rain)

             if count == 0:
                previous_accum_precip = 0.0
                precip_rate.append(0.0)
             else:
                time_delta_sec = (datetime.strptime(current_time, "%Y-%m-%dT%H:%M:%S") - datetime.strptime(prev_time, "%Y-%m-%dT%H:%M:%S")).total_seconds()
                time_delta_hr = time_delta_sec / 3600.0
                previous_accum_precip = accum_precip[-2]
                incremental_rain = total_accum_rain - previous_accum_precip
                rainrate = max(incremental_rain / time_delta_hr, 0.0)
                precip_rate.append(rainrate)


             # Append 2m temperature to T2 list 
             T2.append(float(getvar(wrf_in, 'T2', timeidx=t_idx)[x_y[1],x_y[0]])-273.15)

             # Append mslp to the mslp list
             mslp.append(float(getvar(wrf_in, 'slp', timeidx=t_idx)[x_y[1],x_y[0]]))

             # Read and append cloud fraction for low, mid and high clouds
             cloudfrac = getvar(wrf_in, 'cloudfrac', timeidx=t_idx)[:,x_y[1],x_y[0]].values
             low_cloud_fraction.append(float(cloudfrac[0]))
             mid_cloud_fraction.append(float(cloudfrac[1]))
             high_cloud_fraction.append(float(cloudfrac[2]))

             # Calculate cloud base estimation based on relative humidity profile
             rh = getvar(wrf_in, 'rh', timeidx=t_idx)[:,x_y[1],x_y[0]].values
             z_agl = getvar(wrf_in, 'z', msl=False, timeidx=t_idx)[:,x_y[1],x_y[0]].values

             rh_thresh = 90.0

             cbh_agl = None
             for level in range(len(rh)):
                if cloudfrac[0] > 0.0 or cloudfrac[1] > 0.0:
                   if rh[level] >= rh_thresh:
                      cbh_agl = round(z_agl[level] / 50.0) * 50.0  # already in meters AGL
                      break

             cloud_base.append(cbh_agl)

             # windspeed and direction
             ws = getvar(wrf_in, 'uvmet10_wspd_wdir', units="mph", timeidx=t_idx)[:,x_y[1],x_y[0]].values[0]
             wdir = getvar(wrf_in, 'uvmet10_wspd_wdir', timeidx=t_idx)[:,x_y[1],x_y[0]].values[1]

             windspeed.append(ws)
             winddirection.append(wdir)

             prev_time = current_time
             count = count + 1

   df = pd.DataFrame({
      "temperature": T2,
      "precip_rate": precip_rate,
      "mslp": mslp,
      "cloud_low": low_cloud_fraction,
      "cloud_mid": mid_cloud_fraction,
      "cloud_high": high_cloud_fraction,
      "cloud_base": cloud_base,
      "wind_speed": windspeed,
      "wind_dir": winddirection
   }, index=pd.to_datetime(times))

   # --- Start plot ---
   width = 16
   height = 6
   aspect_ratio = float(width)/float(height)
   fig, ax1 = plt.subplots(figsize=(width, height))

   x_axis_diff = mdates.date2num(df.index[-1]) - mdates.date2num(df.index[0])

   if x_axis_diff*24.0 < float(timeline_length):
       raise ValueError("Warning! Not enough data is available to generate the requested timeline")


# --------------------------------
# 1. Sunrise/Sunset shading
# --------------------------------
   ax1.axvspan(df.index[0], sunrise, color='midnightblue', alpha=0.3)
   ax1.axvspan(sunset, df.index[-1], color='midnightblue', alpha=0.3)
   ax1.axvspan(sunrise,sunset, color='yellow', alpha= 0.1)

# --------------------------------
# 2. CLOUD COVER STACK (top shading)
# --------------------------------
   cloud_total = df["cloud_low"] + df["cloud_mid"] + df["cloud_high"]
   top_level = 35  # degrees C top, use same limit as T2 axis
   bot_level = -5
   y_axis_diff = top_level - bot_level

# Normalize cloud fraction (0-100%) to max shading height
   cloud_low = df["cloud_low"] / 10  * top_level
   cloud_mid = df["cloud_mid"] / 10 * top_level
   cloud_high = df["cloud_high"] / 10  * top_level

# Stack from top downward
   ax1.fill_between(df.index, top_level-5 + cloud_mid/2.0 , top_level-5 + cloud_mid/2.0 + cloud_high, color='silver', alpha=0.3, label="High Cloud")
   ax1.fill_between(df.index, top_level-5 + cloud_mid/2.0, top_level-5 - cloud_mid/2.0, color='gray', alpha=0.4, label="Mid Cloud")
   ax1.fill_between(df.index, top_level-5 - cloud_mid/2.0 - cloud_low, top_level-5 - cloud_mid/2.0, color='dimgray', alpha=0.5, label="Low Cloud")

# --------------------------------
# 3. Temperature
# --------------------------------
   ax1.plot(df.index, df["temperature"], color='red', linewidth=3, label="2m Temperature (°C)")

#--------------------------------
# 4. Windspeed
# --------------------------------
   ax1.plot(df.index, df["wind_speed"], color='green', linewidth=3, label="10 m windspeed (mph)")

# --------------------------------
# 4. Precipitation bars
# --------------------------------
   ax1.bar(df.index, df["precip_rate"], width=0.041666, color='cornflowerblue', alpha=0.6, label="Precip Rate (mm/h)")

# --------------------------------
# 5. Wind arrows
# --------------------------------
   for i, time in enumerate(df.index):
      u = df["wind_speed"].iloc[i] * np.sin(np.radians(df["wind_dir"].iloc[i]))
      v = df["wind_speed"].iloc[i] * np.cos(np.radians(df["wind_dir"].iloc[i]))

      X = mdates.date2num(time)
      Y = -1.5
      
      ax1.quiver(X, Y, u*x_axis_diff, v*y_axis_diff*aspect_ratio, scale=500, width=0.003, angles='xy', scale_units='xy', color='k', zorder=5, pivot='middle')

# --------------------------------
# 5. Pressure (right axis)
# --------------------------------

   mslp_min, mslp_max, pressure_locator = get_pressure_axis_settings(df["mslp"])
   ax2 = ax1.twinx()
   ax2.set_ylim(mslp_min,mslp_max)
   ax2.plot(df.index, df["mslp"], color='black', linewidth=2, label="MSLP (hPa)")
   ax2.set_ylabel("MSLP (hPa)", color='black', fontsize=12)
   ax2.yaxis.set_major_locator(pressure_locator)
   ax2.tick_params(axis='y', labelcolor='black')

# --------------------------------
# 7. Formatting
# --------------------------------
   temp_color = 'tab:red'
   precip_color = 'tab:blue'
   wind_color = 'tab:green'

   # Add color-coded text next to y-axis
   ax1.text(-0.05, 0.5, "Temperature (°C)", color='tab:red',
         transform=ax1.transAxes, rotation=90,
         va='center', ha='center', fontsize=12)

   ax1.text(-0.03, 0.25, "Precip Rate (mm/h)", color='tab:blue',
         transform=ax1.transAxes, rotation=90,
         va='center', ha='center', fontsize=12)

   ax1.text(-0.03, 0.75, "Wind Speed (mph)", color='tab:green',
         transform=ax1.transAxes, rotation=90,
         va='center', ha='center', fontsize=12)

   ax1.set_xlim(df.index[0], df.index[-1])  # limit to data only
   ax1.set_ylim(-5.0, 35.0)
   ax1.xaxis.set_major_locator(HourLocator(interval=3))
   ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))

   ax1.grid(True, linestyle='--', alpha=0.4)
   fig.autofmt_xdate()

# --------------------------------
# 8. Legend
# --------------------------------

   lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
   lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

   # Separate cloud-related entries
   cloud_labels = ["Low Cloud", "Mid Cloud", "High Cloud"]
   cloud_entries = [(h, l) for h, l in zip(lines, labels) if l in cloud_labels]
   other_entries = [(h, l) for h, l in zip(lines, labels) if l not in cloud_labels]

   cloud_handles, cloud_labels = zip(*cloud_entries)
   other_handles, other_labels = zip(*other_entries)

   # First legend: Cloud cover column (left-top)
   legend1 = plt.legend(cloud_handles, cloud_labels,
                     loc='upper right',
                     bbox_to_anchor=(1.01, 1.15),  # You can adjust this
                     frameon=False,
                     fontsize=10)

   # Second legend: everything else in a row (center-top)
   legend2 = plt.legend(other_handles, other_labels,
                     loc='upper center',
                     bbox_to_anchor=(0.75, 1.15),
                     ncol=2,  # Adjust based on number of items
                     frameon=False,
                     fontsize=10)

   # Add both legends to the plot
   plt.gca().add_artist(legend1)

   ax1.set_title('Meteogram '+location_name+'\nLat: '+'{:.2f}'.format(float(str(timeline_lat[0])))+'   Lon: ''{:.2f}'.format(float(str(timeline_lon[0])))+'\nSimulation start time: '+sim_start_time['SIMULATION_START_DATE']+', Valid times: '+str(timeline_starttime)+' to '+str(timeline_endtime)+'\nSunrise='+ sunrise.strftime("%H:%M")+', Sunset='+ sunset.strftime("%H:%M"), horizontalalignment='left', fontsize=10, fontweight='bold', loc='left')

   fig.subplots_adjust(top=0.82)

   plt.tight_layout()

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
   dest_dir = sys.argv[2]
   if not os.path.isdir(dest_dir):
       os.makedirs(dest_dir)

# define lat, lon and location_id lists
   lat = []
   lon = []
   location_id = []
   timeline_starttime = []
   timeline_length = []

# Input WRF out file as an argument (full path)
   wrf_fil = sys.argv[1]
   base_wrf_fil = os.path.basename(wrf_fil)
   dom = base_wrf_fil.split("_")[1]
   date = base_wrf_fil.split("_")[2]
   time = base_wrf_fil.split("_")[3].replace(":", "-")


# Input location information

   lat.append(sys.argv[3])
   lon.append(sys.argv[4])
   location_id.append(sys.argv[5])
   timeline_starttime.append(sys.argv[6])
   timeline_length.append(sys.argv[7])

# Create input dictionary timeline and pass to timeline_meteogram function above

   input_dict = {}
   input_dict["latitudes"] = [float(lat[0])]
   input_dict["longitudes"] = [float(lon[0])]
   input_dict["leveltype"] = "altitude"
   input_dict["infile"] = wrf_fil
   input_dict["locationname"] = location_id[0]
   input_dict["timelinestarttime"] = timeline_starttime[0]
   input_dict["timelinelength"] = timeline_length[0]

   fig = timeline_meteogram(input_dict)

   if fig:
 
      plt.savefig(dest_dir+"/meteogram_timeline_"+dom+"_"+date+"_"+time+"_"+location_id[0]+".png", bbox_inches='tight')


