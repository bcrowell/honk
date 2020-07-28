import math
from pie import Pie
import numpy,scipy

def generate(fc,t,rate1,rate2,width,shape_r,shape_w):
  """
  fc = central frequency of the note
  t = length in seconds of the vibrato envelope to be generated
  rate1 = rate of the initial and final parts of the vibrato, in cycles/s
  rate2 = rate of the main part of the vibrato, in cycles/s
  width = full width of the main part, in semitones (not half-width, so width=2 means +-1 semitone)
  shape_r = four times in units of 1/rate2; first two are measured from the front of the note and give the delay until the vib starts
        and the time for it to start up; final two are similar but measured from the end; these modulate the rate
  shape_w = similar, but for width
  In principle, we want f = fc + env_w cos int 2pi env_r dt. Instead, to make this more tractable to put into polynomail form,
  locate the points where int env_r dt = n pi, which are approximately the extrema of f. Connect each extremum to the next
  with a clamped cubic spline.
  We never want env_r to be zero, because then f is constant and not equal to fc, so it just sounds like we're playing the
  note out of tune. This is the reason for making rate1 nonzero. It's OK to make rate1 equal to rate2, in which case the
  vib doesn't accelerate or decelerate.
  """
  tv = 1.0/rate2 # time for one vibrato cycle
  df = (2.0**(0.5*width/12.0)-1)*fc
  env_r = Pie.window([0,shape_r[0]*tv,shape_r[1]*tv,t-shape_r[2]*tv,t-shape_r[3]*tv,t],rate1,rate2)
  env_w = Pie.window([0,shape_w[0]*tv,shape_w[1]*tv,t-shape_w[2]*tv,t-shape_w[3]*tv,t],0,df)
  max_n = 2*(int(t/tv)+3) # conservative upper estimate of how many vib cycles we can have
  phase = env_r.antiderivative().scalar_mult(2.0*math.pi)
  tn = [] # array containing times of (approximated) extrema
  for n in range(max_n):
    roots = phase.solve(n*math.pi)
    if len(tn)>0 and len(roots)==0:
      continue
    if len(roots)>1:
      raise Exception("non-unique root for phase in vibrato")
    te = roots[0] # time of nth extremum
    if te>t:
      continue
    tn.append(te)
  fn = [] # array containing frequencies at extrema
  for n in range(len(tn)):
    te = tn[n] # time of nth extremum  
    if n%2==0:
      sgn = 1.0
    else:
      sgn = -1.0
    fn.append(fc+env_w(te)*sgn)
  print("tn=",tn)
  #print(env_w(1.5))
  print("fn=",fn)

# test code, to be removed later

generate(100,3,5,7,1,[3,5,2,1],[3,5,2,1])

  
