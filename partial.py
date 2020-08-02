import math,copy
import pie
from pie import Pie

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

  def filter(self,filt):
    """
    Do a sort of mock-up of a filter, using the function filt that takes a frequency as an input and gives a (real-valued) gain as an output.
    This is not linear, and is not really what people think of when they say "filter" in DSP. It also doesn't act on phases. It's
    designed to fit my model of synthesis and to try to do the same thing perceptually as a normal "filter."
    """
    modulation = []
    for t in self.f.x:
      f = self.f(t)
      gain = filt(f)
      #print(f"f={f}, gain={gain}")
      modulation.append(gain)
    am = Pie.join_extrema(self.f.x,modulation)
    # If knots of frequency are extrema (as expected with FM constructed by my method for vibrato), then these are also almost certainly
    # extrema of gain. There could actually be more extrema in between that we miss, as when a violin vib runs back and forth over multiple resonances
    # in a high-frequency partial.
    self.a = self.a.approx_product(am)
