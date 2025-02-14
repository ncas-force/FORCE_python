def plotter(x):
   import importlib

   plottype_content = x["plottype"] + "_" + x["content"]
   print("Calling "+plottype_content+" module")

# Import the appropriate function and set this as plotting_function
   plotting_function = eval(f"__import__(plottype_content).{plottype_content}")

# Use plotting function to create fig
   fig = plotting_function(x)

# Return fig to the code that called the plotter function
   return(fig)

if __name__ == "__main__":
   import matplotlib.pyplot as plt    

# Tests to make sure parsing of dictionaries is functioning properly.

   command_dict = {"plottype":"crosssection", "latitudes":[58.0, 58.25], "longitudes":[-5.25, -5.375], "infile":"/home/earajr/cul_mor/output/CulMor/Scotland/data/2024102200/wrfout_d03_2024-10-22_15:06:00", "content":"specifichumidity", "windvector":True, "leveltype":"altitude", "levels": [0, 4000], "times":["20241022T150600"]}
#   command_dict = {"plottype":"profile", "latitudes":[53.5], "longitudes":[0.5], "infile":"/home/force-nwr/nwr/uk/data/2023041400/wrfout_d02_2023-04-14_14:00:00", "content":"skewT"}
#   command_dict = {"plottype":"map", "latitudes":[0, 0], "longitudes":[0, 0], "leveltype":"pressure", "levels": [850], "infile":"/home/earajr/cul_mor/output/wrfout_d03_2024-10-22_15:06:00", "content":"w"}

   fig = plotter(command_dict)
   plt.savefig("test_figure.png", bbox_inches='tight')


######################################################################
#KEEP THIS FOR AUTO FILLING DEFAULT VALUES FOR ROUTINE PLOT GENERATION
######################################################################

#def plotter(x):
#
#   if x['plottype'] == "crosssection":
#      crosssection_plotter(x)
#   elif x['plottype'] == "profile":
#      profile_plotter(x)
#   elif x['plottype'] == "map":
#      print("Its a map")
#   elif x['plottype'] == "timeseries":
#      print("Its a timeseries")
#   elif x['plottype'] == "trajectory":
#      print("Its a trajectory")
#
#def crosssection_plotter(x):
#   if "locationname" not in x:
#      x["locationname"] = " "
#   if "leveltype" not in x:
#      x["leveltype"] = "altitude"
#      x["levels"] = [0.0, 10000.0]
#   if "alts" not in x:
#      x["alts"] = [0]*len(x["latitudes"])
#   if "windvector" not in x:
#      x["windvector"] = "False"
#
#   plottype_content = x["plottype"] + "_" + x["content"]
#
#   import importlib
#   plotting_function = eval(f"__import__(plottype_content).{plottype_content}")
#
#   fig = plotting_function(x)
#
#   import matplotlib.pyplot as plt
#   plt.savefig("test_crosssection.png", bbox_inches='tight')
#
#
#def profile_plotter(x):
#   if "locationname" not in x:
#      x["locationname"] = " "
#
#   plottype_content = x["plottype"] + "_" + x["content"]
#   print(plottype_content)
#
#   import importlib
#   plotting_function = eval(f"__import__(plottype_content).{plottype_content}")
#
#   fig = plotting_function(x)
#
#   import matplotlib.pyplot as plt
#   plt.savefig("test_profile.png", bbox_inches='tight')
#


