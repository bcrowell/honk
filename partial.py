import math,copy
import pie

class Partial:
  def __init__(self,f,a):
    """
    f and a are Pie objects
    """
    self.a = a
    self.f = f # leave a copy here for things like graphing and debugging, but this is not actually used in computations; also used in restrict()
    self.phi = f.scalar_mult(math.pi*2.0).antiderivative()
    # ... Convert cycles/s to radians/s, then integrate. This should result in a fourth-order polynomial representing the phase phi(t).
    if self.a.order()!=3:
      raise Exception(f"in constructor for Partial, amplitude is not a cubic polynomial, has order {self.a.order()} instead")
    if self.phi.order()!=4:
      raise Exception(f"in constructor for Partial, phase is not a quartic polynomial, has order {self.phi.order()} instead")

  @classmethod
  def from_phase_and_amplitude(cls,phi,a):
    """
    This is required when we need to get the absolute phase right, as when splitting a time interval into two parts without a discontinuity.
    """
    result = cls(phi.derivative(),a) # initialize it with the wrong constant of integration
    result.phi = phi # fix incorrect constant of integration
    return result

  def time_range(self):
    return self.a.time_intersection(self.phi)

  def scale_f(self,s):
    result = copy.deepcopy(self)
    result.phi = result.phi.scalar_mult(s)
    result.f = result.f.scalar_mult(s)
    return result

  def scale_a(self,s):
    result = copy.deepcopy(self)
    result.a = result.a.scalar_mult(s)
    return result

  def restrict(self,t1,t2):
    return Partial.from_phase_and_amplitude(self.phi.restrict(t1,t2),self.a.restrict(t1,t2))