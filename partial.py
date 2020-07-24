import math

class Partial:
  def __init__(self,f_times,f_values,a_times,a_values,phase):
    self.omega_times = f_times
    self.omega_values = list(map(lambda f:2.0*math.pi*f,f_values)) # convert from cycles/s to radians/s
    print("self.omega_times = ",self.omega_times)
    print("self.omega_values = ",self.omega_values)
    self.a_times = a_times
    self.a_values = a_values
    self.phase = phase
