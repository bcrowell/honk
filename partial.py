import math,copy
import pie

class Partial:
  def __init__(self,f,a,phase):
    """
    f and a are Pie objects
    """
    self.a = a
    self.omega = f.scalar_mult(math.pi*2.0) # convert cycles/s to radians/s
    self.phase = phase

  def time_range(self):
    return self.a.time_intersection(self.omega)

  def scale_f(self,s):
    result = copy.deepcopy(self)
    result.omega = result.omega.scalar_mult(s)
    return result

  def scale_a(self,s):
    result = copy.deepcopy(self)
    result.a = result.a.scalar_mult(s)
    return result
