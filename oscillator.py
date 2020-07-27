import numpy,functools,scipy,math
from scipy import interpolate

class Oscillator:
  def __init__(self,n_samples,max_spline_knots,spline_order,max_spline_coeffs,max_partials):
    self.max_spline_knots = max_spline_knots
    self.spline_order = spline_order
    self.max_spline_coeffs = max_spline_coeffs
    self.max_partials = max_partials
    # buffer to hold synthesized sound:
    self.y = numpy.zeros(n_samples, numpy.float32)
    # misc data structures:
    self.clear_small_arrays()

  def error_code(self):
    return self.err[0]

  def clear(self):
    self.y.fill(0.0)
    self.clear_small_arrays()

  def clear_small_arrays(self):
    self.err = numpy.zeros(1, numpy.int32)
    self.info = numpy.zeros(100, numpy.float32)
    self.n_info = numpy.zeros(1, numpy.int32)
    self.phi_c = numpy.zeros(self.max_spline_coeffs, numpy.float32)
    self.phi_knots = numpy.zeros(self.max_spline_knots, numpy.float32)
    self.a_c = numpy.zeros(self.max_spline_coeffs, numpy.float32)
    self.a_knots = numpy.zeros(self.max_spline_knots, numpy.float32)
    self.phi_n = numpy.zeros(self.max_partials, numpy.int32)
    self.a_n = numpy.zeros(self.max_partials, numpy.int32)
    self.i_pars = numpy.zeros(100, numpy.int64)
    self.f_pars = numpy.zeros(100, numpy.float32)

  def setup(self,partials):
    self.clear()
    self.partials = partials
    if len(functools.reduce(cat,list(map(lambda p:p.phi.x,partials))))>self.max_spline_knots:
      raise Exception("too many phi knots")
    print("n phi knots = ",len(functools.reduce(cat,list(map(lambda p:p.phi.x,partials))))) # qwe
    # create flattened versions of input data for consumption by opencl
    two_pi = 2.0*math.pi
    copy_into_numpy_array(self.phi_knots,     functools.reduce(cat,list(map(lambda p:p.phi.x,partials))) )
    copy_into_numpy_array(self.a_knots,       functools.reduce(cat,list(map(lambda p:p.a.x,    partials))) )
    copy_into_numpy_array(self.phi_c,         functools.reduce(cat,list(map(lambda p:p.phi.c.flatten(),partials))) )
    copy_into_numpy_array(self.a_c,           functools.reduce(cat,list(map(lambda p:p.a.c.flatten(),    partials))) )
    for i in range(len(partials)):
      p = partials[i]
      self.phi_n[i] = len(p.phi.x)
      self.a_n[i] = len(p.a.x)
      print("partial ",i,", n phi knots=",self.phi_n[i]) # qwe
    self.i_pars[1] = len(partials)

  def time_range(self):
    a,b = self.partials[0].time_range()
    for p in self.partials:
      (aa,bb) = p.time_range()
      a = max(a,aa)
      b = min(b,bb)
    return (a,b)    

  def in_time_range(self,t):
    a,b = self.time_range()
    return (t>=a and t<=b)

  def __str__(self):
    result = ''
    result = result + "valid time range = "+str(self.time_range())+"\n"
    result = result + "a_knots = "+sa(self.a_knots)+"\n"
    result = result + "a_c = "+sa(self.a_c)+"\n"
    return result

def sa(a):
  # make an array into a string, omitting trailing zeroes
  last_nonzero = 0
  for i in range(len(a)):
    if a[i]!=0:
      last_nonzero = i
  return str(a[0:last_nonzero+1])

def copy_into_numpy_array(x,y):
  for i in range(len(y)):
    x[i] = y[i]

# concatenate two lists
def cat(l1,l2):
  return [*l1,*l2]
