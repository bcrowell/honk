import numpy

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

