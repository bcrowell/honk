import numpy

class Oscillator:
  def __init__(self,n_samples,max_spline_knots,spline_order,max_spline_coeffs,max_partials):
    self.y = numpy.zeros(n_samples, numpy.float32)
    self.err = numpy.zeros(1, numpy.int32)
    self.info = numpy.zeros(100, numpy.float32)
    self.n_info = numpy.zeros(1, numpy.int32)
    self.omega_c = numpy.zeros(max_spline_coeffs, numpy.float32)
    self.omega_knots = numpy.zeros(max_spline_knots, numpy.float32)
    self.a_c = numpy.zeros(max_spline_coeffs, numpy.float32)
    self.a_knots = numpy.zeros(max_spline_knots, numpy.float32)
    self.phase = numpy.zeros(max_partials, numpy.float32)
    self.omega_n = numpy.zeros(max_partials, numpy.int32)
    self.a_n = numpy.zeros(max_partials, numpy.int32)
    self.i_pars = numpy.zeros(100, numpy.int64)
    self.f_pars = numpy.zeros(100, numpy.float32)

  def error_code(self):
    return self.err[0]
