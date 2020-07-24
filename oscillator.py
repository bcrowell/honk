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
    self.omega_c = numpy.zeros(self.max_spline_coeffs, numpy.float32)
    self.omega_knots = numpy.zeros(self.max_spline_knots, numpy.float32)
    self.a_c = numpy.zeros(self.max_spline_coeffs, numpy.float32)
    self.a_knots = numpy.zeros(self.max_spline_knots, numpy.float32)
    self.phase = numpy.zeros(self.max_partials, numpy.float32)
    self.omega_n = numpy.zeros(self.max_partials, numpy.int32)
    self.a_n = numpy.zeros(self.max_partials, numpy.int32)
    self.i_pars = numpy.zeros(100, numpy.int64)
    self.f_pars = numpy.zeros(100, numpy.float32)

  def setup(self,partials):
    self.clear()
    # create flattened versions of input data for consumption by opencl
    two_pi = 2.0*math.pi
    copy_into_numpy_array(self.omega_knots,   functools.reduce(cat,list(map(lambda p:p.omega_times,partials))) )
    copy_into_numpy_array(self.a_knots,       functools.reduce(cat,list(map(lambda p:p.a_times,partials))) )
    copy_into_numpy_array(self.phase,         [p.phase for p in partials] )
    copy_into_numpy_array(self.omega_c,       functools.reduce(cat,list(map(lambda p:cubic_spline_coeffs(p.omega_times,p.omega_values),partials))) )
    copy_into_numpy_array(self.a_c,           functools.reduce(cat,list(map(lambda p:cubic_spline_coeffs(p.a_times,p.a_values),partials))) )
    for i in range(len(partials)):
      p = partials[i]
      self.omega_n[i] = len(p.omega_times)
      self.a_n[i] = len(p.a_times)
    self.i_pars[1] = len(partials)

  def __str__(self):
    return str(self.omega_c)

def copy_into_numpy_array(x,y):
  for i in range(len(y)):
    x[i] = y[i]

def cubic_spline_coeffs(x,y):
  return scipy.interpolate.CubicSpline(x,y).c.flatten()

# concatenate two lists
def cat(l1,l2):
  return [*l1,*l2]
