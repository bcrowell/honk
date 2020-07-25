# a subclass of scipy's PPoly class with some extra features

import copy,re
import scipy
from scipy import interpolate
from scipy.interpolate import PPoly

class Pie(PPoly):
  def __init__(self,p):
    # p is a PPoly object
    self.c = p.c
    self.x = p.x
    self.axis = p.axis # not used

  @classmethod
  def from_string(cls,s,last_pair=None):
    """
    Constructor that takes an input in the format "t v,t v ... ; t v,t v ... ; ...".
    The ; characters separate groups, each of which is a cubic spline.
    Putting c at the end of a group makes it "clamped," as described in the docs for scipy.interpolate.CubicSpline, i.e., deriv=0 at ends.
    If the first t v pair in a group is just whitespace, it gets copied from the end of the previous group, causing the function
    to be continuous.
    """
    if re.search(";",s):
      l = re.split(r"\s*;\s*", s)
      result = Pie.from_string(l[0])
      for sub in l[1:]:
        result = result.cat(Pie.from_string(sub,result.__last_pair))
      return result
    bc = 'not-a-knot'
    if re.search("c",s):
      bc = 'clamped'
    t = []
    v = []
    for sub in re.split(r"\s*,\s*", s):
      if re.search("[^\s]",sub):
        capture = re.search(r"([^\s]*)\s+([^c\s]*)",sub)
        tt,vv = capture.group(1,2)
      else:
        tt,vv = last_pair
      t.append(float(tt))
      v.append(float(vv))
    result = Pie(scipy.interpolate.CubicSpline(t,v,bc_type=bc))
    result.__last_pair = [t[-1],v[-1]] # for use within constructor
    return result

  def scalar_mult(self,s):
    r = copy.deepcopy(self)
    for i in range(len(r.c)):
      for j in range(len(r.c[i])):
        r.c[i][j] = s*r.c[i][j]
    return r

  def time_range(self):
    return (self.x[0],self.x[-1])

  def time_intersection(self,q):
    """
    p.time_intersection(q) gives the intersection of the domains of p and q
    """
    a = self.time_range()
    b = q.time_range()
    return (max(a[0],b[0]),min(a[-1],b[-1]))

  def cat(self,q):
    """
    Concatenate two Pie objects that have an endpoint in common. Returns the result. Doesn't change the input.
    Raises an exception if they don't have an endpoint in common.
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
    
