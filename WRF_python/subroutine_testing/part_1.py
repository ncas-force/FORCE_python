import glob
from multiprocessing.pool import Pool
import sys

def command_generator(x): # This function takes a dictionary as an argument and then uses the required parts to create the plotting commands. As the command generator is run in parallel there is no need to invoke GNU parallel. The advantage of this approach is that you can add additional eleemnts to the command_list dictionary and it will not cause problems in the generation of images as the command generator will select only the required keys to produce the command to be run. Obviously the generation of the sounding command has yet to be written and the structure of the actual xsection command is very flexible within the command generator (while still accepting a dictionary of any size as input). 
   import subprocess
   import os

   if x["plot_type"] == "xsection":
      print("Creating cross section")
      command_string = "python "+x["plotting_script"]+" "+x["bespoke_flag"]+" "+x["lats"]+" "+x["lons"]+" "+x["base_alt"]+" "+x["top_alt"]+" "+x["input_dir"]+" "+x["output_dir"]+" "+x["wrf_in"]
#      subprocess.run(command_string, shell=True)
      print(command_string)
      
   if x["plot_type"] == "sounding":
      print("Creating sounding")


# This section replicates part of Xesc.py that currently manages generation of commands as such some filler values are created below.

input_dir = "/home/shared/nwr/uk/data/2023030800"
output_dir = "/home/earajr/FORCE_WRF_plotting/WRF_python/subroutine_testing/output"
plotting_script = "/home/earajr/FORCE_WRF_plotting/WRF_python/theta_w_xsection.py"
plot_type = "xsection"
lats="51.0,51.0"
lons="-3.7,-1.6"
base_alt="0"
top_alt="10000"
n_cores = 4

# Empty command_list produced to be populated by dictionaries of arguments
command_list = []

# Use glob.glob to identify wrf_in files and append dictionaries to command_list
for wrf_in in glob.glob(input_dir+"/wrfout_d02*"):
    command_list.append({"input_dir":input_dir, "output_dir":output_dir, "plotting_script":plotting_script, "plot_type":plot_type, "wrf_in":wrf_in, "lats":lats, "lons":lons, "base_alt":base_alt, "top_alt":top_alt, "bespoke_flag":"1"})

# As n_cores is set to 4 above the command_generator function willbe invoked in parallel for all dictionaries held in the command_list
if n_cores > 1:
   print("--pooling starts now--")
   pool = Pool( processes=n_cores )
   r = pool.map_async(command_generator, command_list)
   r.wait() # Wait for the results
   print("--pooling ended--")
   if not r.successful():
      print(r._value, sys.exit("Parallelization not successful"))
