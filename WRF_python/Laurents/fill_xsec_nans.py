def fill_xsec_nans(x):
   import numpy as np
  
   for i in np.arange(0, np.shape(x)[1], 1):
      ind = np.where(~np.isnan(x[:,i]))[0]
      first = ind[0]
      x[:first,i] = x[first,i]

   return x
