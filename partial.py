import math,copy

class Partial:
  def __init__(self,f,a,phase):
    """
    f and a are scipy PPoly objects
    """
    self.a = a
    self.omega = copy.deepcopy(f)
    for i in range(len(self.omega.c)):
      for j in range(len(self.omega.c[i])):
        self.omega.c[i][j] = math.pi*2.0*self.omega.c[i][j] # convert cycles/s to radians/s
    self.phase = phase

  def time_range(self):
    return (max(self.a.x[0],self.omega.x[0]),min(self.a.x[-1],self.omega.x[-1]))

