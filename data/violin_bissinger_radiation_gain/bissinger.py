"""
Data from Bissinger, 2002, Radiation damping, efficiency, and directivity for violin normal modes below 4 kHz.
Values of R/Y from fig. 1.
R = sound pressure/force, radiativity (not a widely used term for this)
Y = mobility=velocity/force
This measures how well different modes radiate, is an rms amplitude measurement, a ratio of sound pressure to force.
The main effect of this is to include the main air resonance.
Bissinger has data in absolute SI units, but I only attempted to extract relative values.
"""

import math
import scipy.interpolate
import data.violin_bissinger_radiation_gain.bissinger as bissinger

def radiation_gain(f):
  """
  Input is frequency in Hz. Output is a gain factor (linear, not db).
  """
  if not hasattr(radiation_gain,"func"):
    x = []
    y = []
    pts = ry_list()
    for p in pts:
      xx,yy = p
      x.append(xx)
      y.append(yy)
    radiation_gain.func = scipy.interpolate.interp1d(
            x,y,
            fill_value=(0.5581,1.0465),
            bounds_error=False) # default is linear
    # https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html#d-interpolation-interp1d
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
  a = radiation_gain.func(f)-1.0 # arbitrarily add a constant so gains are close to unity.
  gain = 10.0**a
  max_f = 980.0 # maximum freq for which I have data
  cutoff = 4500.0
  if f<max_f:
    return gain
  if f>max_f and f<cutoff:
    return gain*math.sqrt(f/max_f) # not sure if this is right
  # fall through, f>=cutoff
  return gain*math.sqrt(cutoff/max_f) # not sure if this is right

def ry_list():
  """
  X axis is frequency in Hz.
  Y axis is log10(R/Y).
  """
  # f (Hz),log10(R/Y)
  return [
[220,0.558139534883721],
[240,0.837209302325581],
[260,1.18604651162791],
[270,1.53488372093023],
[295,1.6046511627907],
[340,1.2906976744186],
[400,1.11627906976744],
[440,0.906976744186046],
[460,0.906976744186046],
[540,0.906976744186046],
[615,0.976744186046512],
[700,1.25581395348837],
[840,1.11627906976744],
[920,1.18604651162791],
[980,1.04651162790698]
  ]

