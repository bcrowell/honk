"""
A subclass of scipy's PPoly class with some extra features:
  - a convenience method for creating it from a string describing a series of splines
  - read out the order of the polynomial
  - scalar multiplication
  - time range and intersections of time ranges
  - cat and join methods
  - graphing
  - window
  - join_extrema
"""


import copy,re
import scipy,numpy
from scipy import interpolate
from scipy.interpolate import PPoly
import matplotlib
import matplotlib.pyplot as plt

class Pie(PPoly):
  # construct it from a PPoly object, e.g., Pie(scipy.interpolate.CubicSpline([...],[...],bc_type=bc))
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
    self.assert_valid()
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
    result.x = result.x[lo_i:hi_i+1] # python's slice method is defined such that this copies indices lo_i through hi_i
    # now make c the desired thing, which means it covers indices lo_i through hi_i-1
    new_c = []
    for row in result.c:
      new_row = row[lo_i:hi_i]
      new_c.append(new_row)
    result.c = numpy.asarray(new_c, dtype=numpy.float)
    result.assert_valid()
    return result

  def assert_valid(self):
    if self.x.shape[0]!=self.c.shape[1]+1:
      raise Exception(f'shapes of x and c in {self} are not compatible, should have self.x.shape[0]=self.c.shape[1]+1')

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

  @classmethod
  def window(cls,t,y1,y2):
    """
    Construct a "window" function analogous to things like Hamming windows.
    t[] should have 6 elements.
    """
    y = [y1,y1,y2,y2,y1,y1]
    return Pie.join_extrema(t,y)

  @classmethod
  def join_extrema(cls,t,y):
    """
    Inputs data points (t1,y1), (t2,y2), ..., which are taken as extrema of a smooth curve.
    Outputs a piecewise polynomial consisting of clamped cubic splines connecting these points.
    """
    p = []
    for i in range(len(t)-1):
      p.append(Pie(scipy.interpolate.CubicSpline([t[i],t[i+1]],[y[i],y[i+1]],bc_type='clamped')))
    return Pie.join(p)

  def approx_product(self,q,min_h=0.001):
    return Pie.arith(self,q,min_h,op='*')

  def sum(self,q,min_h=0.001):
    return Pie.arith(self,q,min_h,op='+')

  def arith(self,q,min_h=0.001,op='*'):
    """
    Add or multiply two piecewise cubics. In the case of multiplication, we can't get an exact result without
    increasing the order, so we instead try to form an approximation to the product using a piecewise cubic.
    The knots of the result are the union of the knots of the two functions.
    The polynomial in each interval is built to have the correct values and derivatives at the end-points of that interval,
    which in the case of addition gives an exact result.
    When two knots lie within min_h of each other, we delete the later one.
    """
    u = set(self.x).union(set(q.x)) # union of knots of the two functions
    l = sorted(list(u))
    # Get rid of very short pieces.
    new_l = [l[0]]
    for i in range(len(l)-1):
      if abs(new_l[-1]-l[i+1])>min_h:
        new_l.append(l[i+1])
    new_x = numpy.asarray(new_l,dtype=numpy.float) 
    pd = self.derivative()
    qd = q.derivative()
    polys = []
    for i in range(len(new_x)-1):
      x1 = new_x[i]
      x2 = new_x[i+1]
      h = x2-x1
      eps = h*1.0e-5
      # Evaluate P, Q, P', and Q' at the end-points.
      p1 = self(x1)
      p2 = self(x2)
      q1 = q(x1)
      q2 = q(x2)
      p1d = pd(x1+eps)
      p2d = pd(x2-eps)
      q1d = qd(x1+eps)
      q2d = qd(x2-eps)
      if op=='+':
        # Define R(x)=P(x)+Q(x).
        r1 = p1+q1
        r2 = p2+q2
        r1d = p1d+q1d
        r2d = p2d+q2d
      if op=='*':
        # Define R(x)=P(x)Q(x).
        r1 = p1*q1
        r2 = p2*q2
        # Use Leibniz rule to evaluate R and R' at the end-points.
        r1d = p1*q1d+p1d*q1
        r2d = p2*q2d+p2d*q2
      # Find a+bx+cx^2+dx^3 that matches R and R' at the endpoints.
      j,k,l,m = Pie.invert_2x2(h**2,h**3,2*h,3*h**2)
      a = r1
      b = r1d
      d1 = r2-a-b*h
      d2 = r2d-b
      c = j*d1+k*d2
      d = l*d1+m*d2
      polys.append([a,b,c,d])
    result = copy.deepcopy(self)
    result.x = new_x
    # flip and twist coeffs into the weird form used by PPoly
    new_c = []
    for row_num in range(4):
      new_row = []
      for col_num in range(len(polys)):
        new_row.append(polys[col_num][3-row_num])
      new_c.append(new_row)
    result.c = numpy.asarray(new_c, dtype=numpy.float)
    result.assert_valid()
    return result

  @staticmethod
  def invert_2x2(a,b,c,d):
    det = a*d-b*c
    return (d/det,-b/det,-c/det,a/det)

