# a subclass of scipy's PPoly class with some extra features

import scipy
from scipy import interpolate
from scipy.interpolate import PPoly

class Pie(PPoly):
  def __init__(self,p):
    # p is a PPoly object
    self.c = p.c
    self.x = p.x
    self.axis = p.axis # not used

  def time_range(self):
    return (self.x[0],self.x[-1])

  def time_intersection(self,q):
    """
    p.time_intersection(q) gives the intersection of the domains of p and q
    """
    a = self.time_range
    b = q.time_range
    return (max(a[0],b[0]),min(a[-1],b[-1]))

