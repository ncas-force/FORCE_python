#####################################################################################################
from metpy.calc import divergence
from metpy.calc import lat_lon_grid_deltas
from numpy import shape

def calc_divergence(x, y, lats, lons):

   dx, dy = lat_lon_grid_deltas(lats, lons)

   x_y_divergence = divergence(x, y, dx=dx, dy=dy)

   return x_y_divergence
 
