import math

class Partial:
  def __init__(self,f_times,f_values,a_times,a_values,phase):
    self.omega_times = f_times
    self.omega_values = list(map(lambda f:2.0*math.pi*f,f_values)) # convert from cycles/s to radians/s
    self.a_times = a_times
    self.a_values = a_values
    self.phase = phase

  def time_range(self):
    return (max(self.a_times[0],self.omega_times[0]),min(self.a_times[-1],self.omega_times[-1]))
