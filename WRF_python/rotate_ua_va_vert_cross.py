def rotate(ua_vertcross, va_vertcross):
   from math import acos, degrees, pi, cos, sin
   import numpy as np
   from wrf import (to_np)
   from geopy.distance import geodesic
   from geographiclib.geodesic import Geodesic


   '''
    Takes u and v wind component vertcross, rotates them to align with
    transect meteorological direction, from start_point to end_point. 
    '''

   coord_pairs_1 = to_np(ua_vertcross.coords["xy_loc"])
   coord_pairs_2 = to_np(va_vertcross.coords["xy_loc"])
   if (any(coord_pairs_1 != coord_pairs_2)):
      print("u-component and v component does not match")
      return
   coord_pairs = coord_pairs_1
   main_lat = [(x.lat) for x in coord_pairs]
   main_lon = [(x.lon) for x in coord_pairs]

   # Create an emptry transect
   met_dir_transect = []

   point_a = main_lat[0], main_lon[0]
   point_b = main_lat[-1], main_lon[-1]
   point_c = main_lat[0], main_lon[-1]

   A = geodesic(point_a, point_b).km
   B = geodesic(point_b, point_c).km
   C = geodesic(point_c, point_a).km

   if B == 0 or C == 0:
      if main_lat[0] == main_lat[-1]:
         par_wind = ua_vertcross
         perp_wind = va_vertcross
      else:
         par_wind = va_vertcross
         perp_wind = ua_vertcross

   else:

#      degrees_A_C = 90 + degrees(acos((A * A + C * C - B * B)/(2.0 * A * C)))
#      print(degrees_A_C)
#      bearing = Geodesic.WGS84.Inverse(point_a[0], point_a[1], point_b[0], point_b[1])['azi1']
#      met_dir_transect.append(degrees_A_C)

      for point in range(len(main_lat)):
         if point == 0:
            point_a = main_lat[point] - (main_lat[point+1]-main_lat[point])/2.0, main_lon[point] - (main_lon[point+1]-main_lon[point])/2.0
            point_b = main_lat[point] + (main_lat[point+1]-main_lat[point])/2.0, main_lon[point] + (main_lon[point+1]-main_lon[point])/2.0
         elif point != len(main_lat)-1:
            point_a = (main_lat[point-1]+main_lat[point])/2.0, (main_lon[point-1]+main_lon[point])/2.0
            point_b = (main_lat[point]+main_lat[point+1])/2.0, (main_lon[point]+main_lon[point+1])/2.0
         else:
            point_a = main_lat[point]-(main_lat[point]-main_lat[point-1])/2.0, main_lon[point]-(main_lon[point]-main_lon[point-1])/2.0
            point_b = main_lat[point]+(main_lat[point]-main_lat[point-1])/2.0, main_lon[point]+(main_lon[point]-main_lon[point-1])/2.0

#         point_a = main_lat[point-1], main_lon[point-1]
#         point_b = main_lat[point], main_lon[point]
#         point_c = main_lat[point-1], main_lon[point]

         bearing = Geodesic.WGS84.Inverse(point_a[0], point_a[1], point_b[0], point_b[1])['azi1']
         met_dir_transect.append(bearing)

#         A = geodesic(point_a, point_b).km
#         B = geodesic(point_b, point_c).km
#         C = geodesic(point_c, point_a).km
#
#         if B == 0 or C == 0:
#            if main_lat[point-1] == main_lat[point]:
#               degrees_A_B = 90.0
#            else:
#               degrees_A_B = 180.0 
#         else:
#            degrees_A_B = 180 - degrees(acos((A * A + B * B - C * C)/(2.0 * A * B)))
# 
#         met_dir_transect.append(degrees_A_B)


      met_dir_transect_radians = np.radians(met_dir_transect)

      c = np.cos(met_dir_transect_radians)
      s = np.sin(met_dir_transect_radians)

      met_dir_transect_2 = np.array(met_dir_transect, ndmin=1)
      a = met_dir_transect_2/180

      c = [cos(pi*X) for X in a]
      s = [sin(pi*X) for X in a]
      c_tile = np.tile(c, (len(ua_vertcross.vertical), 1))
      s_tile = np.tile(s, (len(ua_vertcross.vertical), 1))
      
#      print("CTILE")
#      print(c_tile)
#      print("STILE")
#      print(s_tile)
      par_wind = (ua_vertcross * s) + (va_vertcross * c)
      perp_wind = (ua_vertcross * c) + (va_vertcross * s)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
   return (par_wind, perp_wind)


