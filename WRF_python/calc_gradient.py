#####################################################################################################
from metpy.calc import gradient
from metpy.calc import geospatial_gradient
from metpy.calc import lat_lon_grid_deltas
from numpy import shape

def calc_gradient(x, lats, lons):

   dx, dy = lat_lon_grid_deltas(lats, lons)

   x_gradient = gradient(x, deltas=(dy, dx))

   return x_gradient
 

def calc_gradient_dx(x, dx):
  
   x_gradient = gradient(x, deltas=(dx, dx))

   return x_gradient
