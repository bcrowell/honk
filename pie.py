# a subclass of scipy's PPoly class with some extra features

import copy
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
    a = self.time_range()
    b = q.time_range()
    return (max(a[0],b[0]),min(a[-1],b[-1]))

  def extend(self,q):
    """
    Concatenate two Pie objects that have an endpoint in common. Returns None if they don't have an endpoint in common.
    """
    r = copy.deepcopy(self)
    # Not sure if I'm interpreting the docs for PPoly.extent() correctly. It looks like they want the initial element of q.x,
    # which duplicates the last element of r.x, to be deleted.
    if abs(r.x[-1]-q.x[0])>3.0e-5: # tolerance slighly longer than one sample at 44.1 kHz
      raise Exception("endpoints do not coincide")
    super(Pie,r).extend(q.c,q.x[1:])
    return r

  def __str__(self):
    result = ''
    result = result + "  x = "+str(self.x)+"\n"
    result = result + "  c = "+str(self.c)+"\n"
    return result
    
