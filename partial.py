import math,copy
import pie

class Partial:
  def __init__(self,f,a,phase):
    """
    f and a are Pie objects
    """
    self.a = a
    self.omega = copy.deepcopy(f)
    for i in range(len(self.omega.c)):
      for j in range(len(self.omega.c[i])):
        self.omega.c[i][j] = math.pi*2.0*self.omega.c[i][j] # convert cycles/s to radians/s
    self.phase = phase

  def time_range(self):
    return self.a.time_intersection(self.omega)


