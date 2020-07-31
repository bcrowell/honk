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
    r = Partial(self.f.restrict(t1,t2),self.a.restrict(t1,t2))
    print(f"in Partial.restrict({t1},{t2}), result has time range {r.time_range()}")
    return Partial(self.f.restrict(t1,t2),self.a.restrict(t1,t2))