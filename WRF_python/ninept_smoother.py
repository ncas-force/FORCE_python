#####################################################################################################
import numpy as np
import sys
from scipy.signal import convolve2d


#  9-point smoother function, required to make 2d fields look better.
#
def smth9(x,p,q):
#
#  Run a 9-point smoother on the 2D numpy.array x using weights
#  p and q.  Return the smoothed array.
#
#
#  Get array dimensions and check on sizes.
#
  ni = x.shape[0]
  nj = x.shape[1]
  if (ni < 3 or nj < 3):
    print("smth9: both array dimensions must be at least three.")
    sys.exit()

#
#  Smooth.
#
  
  q*4.0 + p*5.0

  kernel = np.array([[q, p, q], [p, p, p], [q, p, q]])
  kernel = kernel/np.sum(kernel)

  output = convolve2d(x, kernel, mode='same', boundary='symm')
#
#  Return smoothed array.
#
  return output
