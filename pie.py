"""
A subclass of scipy's PPoly class with some extra features:
  - a convenience method for creating it from a string describing a series of splines
  - read out the order of the polynomial
  - scalar multiplication
  - time range and intersections of time ranges
  - cat and join methods
  - graphing
"""


import copy,re
import scipy
from scipy import interpolate
from scipy.interpolate import PPoly
import matplotlib
import matplotlib.pyplot as plt

class Pie(PPoly):
  def __init__(self,p):
    super(Pie,self).__init__(p.c,p.x)

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
      a = []
      for i in range(len(l)):
        sub = l[i]
        if i==0:
          a.append(Pie.from_string(sub))
        else:
          a.append(Pie.from_string(sub,a[-1].last_pair))
      return Pie.join(a)
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
    result.last_pair = [t[-1],v[-1]] # for use within constructor
    return result

  def eval(self,x):
    return self(x)

  def order(self):
    return len(self.c)-1

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
    Raises an exception if they don't have an endpoint in common. To concatenate more than two Pie objects, use join().
    """
    r = copy.deepcopy(self)
    # Not sure if I'm interpreting the docs for PPoly.extent() correctly. It looks like they want the initial element of q.x,
    # which duplicates the last element of r.x, to be deleted.
    if abs(r.x[-1]-q.x[0])>3.0e-5: # tolerance slighly longer than one sample at 44.1 kHz
      raise Exception(f"endpoints {r.x[-1]} and {q.x[0]} do not coincide")
    super(Pie,r).extend(q.c,q.x[1:])
    return r

  @classmethod
  def join(cls,a):
    """
    Pie.join([p1,p2,...]) returns a Pie made by concatenating p1, p2, ...
    """
    for i in range(len(a)):
      aa = a[i]
      if i==0:
        result = copy.deepcopy(aa)
      else:
        result = result.cat(aa)
    return result

  def restrict(self,t1,t2):
    """
    Create a new Pie object by restricting the range of the t variable to [t1,t2].
    """
    n = len(self.x)
    # by default we need them all:
    lo_i = 0
    hi_i = n-1
    # which don't we need?
    for i in range(n):
      if i+1<=n-1 and t1>self.x[i+1]: # [i,i+1] is too early to be needed
        lo_i = i+1
    for i in range(n-2,-1,-1): # count from n-2 down to 0
      if i>=0 and t2<self.x[i]: # [i,i+1] is too late to be needed
        hi_i = i
    result = copy.deepcopy(self)
    result.x = result.x[lo_i:hi_i+1]
    for i in range(len(self.c)):
      result.c[i] = result.c[i][lo_i:hi_i]
    return result

  def __str__(self):
    result = ''
    result = result + "  x = "+str(self.x)+"\n"
    result = result + "  c = "+str(self.c)+"\n"
    return result
    
  def graph(self,filename,t1,t2,n):
    # https://matplotlib.org/gallery/lines_bars_and_markers/simple_plot.html#sphx-glr-gallery-lines-bars-and-markers-simple-plot-py
    # format inferred from filename's extension, png works
    tt = []
    yy = []
    dt = (t2-t1)/(n-1)
    for i in range(n):
      t = t1+dt*i
      tt.append(t)
      yy.append(self.eval(t))
    fig, ax = plt.subplots()
    ax.plot(tt, yy)
    print(f'graph written to file {filename}')
    fig.savefig(filename)

