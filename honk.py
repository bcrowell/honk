#!/bin/python3

import wave,time,math,sys,copy,random
import pyopencl as cl
from pyopencl import array
import numpy,scipy

from opencl_device import OpenClDevice
from oscillator import Oscillator
from partial import Partial
from pie import Pie
import instruments,vibrato

def main():
  dev = OpenClDevice()
  dev.build('oscillator.cl')
   
  length_sec = 3.0
  n_instances = 256 # larger values cause mysterious behavior that smells like corrupted memory or instances modifying each other's memory
  local_size = 64 # must divide n_instances, and my video card prefers it to be at least 32
  sample_freq = 44100.0
  n_samples = int(length_sec*sample_freq)

  a = instruments.violin_envelope()

  # Experiment with delaying attack in higher partials.
  # Guettler, Looking at starting transients and tone coloring of the bowed string.
  scale_delayed_attacks = 1.0
  delay_attack = [] # delay in seconds
  for i in range(len(a)):
    n=i+1
    # d = delay in milliseconds, estimated from Guettler's fig 13.
    if n==1:
      d=15
      continue
    if n>=2 and n<=3:
      d=0
    if n>=4:
      d = 20+21*math.log(n/4.0)
    delay_attack.append(scale_delayed_attacks*d/1000.0)
      
  vib = vibrato.generate(290,3,3,6,0.4,[3,5,4,1],[1,8,4,1])
  partials = []
  for i in range(len(a)):
    n=i+1 # n=1 for fundamental
    if n==1:
      # p1 = Partial(vib,Pie.from_string("0 0,0.2 0.5 c ; , 2 1 ; , 3 0")) # gradual onset
      d = delay_attack[i]
      s = f"0 0, {d+.001} 0 ; , {d+0.05} 0.5 c ; , 2 1 ; , 3 0"
      p1 = Partial(vib,Pie.from_string(s)) # faster attack
      p = p1
    else:
      p = copy.deepcopy(p1).scale_f(n).scale_a(a[i])
    print(f"i={i} p.time_range={p.time_range()}")
    partials.append(copy.deepcopy(p))


  # Experiment with random frequency modulation on attack.
  fm_amount = 0.100
  for p in partials:
    n_fm = int(10*length_sec)
    t = []
    f = []
    for i in range(n_fm):
      tt = (i/float(n_fm))*length_sec
      tt = tt*0.5
      t.append(tt)
      r = fm_amount*(random.random()-0.5)
      end_attack = 0.05
      if tt>end_attack:
        r = r*math.exp(-(tt-end_attack)/0.1)
      f.append(r*p.f(0))
    fm = Pie.join_extrema(t,f)
    p.f = p.f.sum(fm)
    p.phi = p.f.scalar_mult(math.pi*2.0).antiderivative()

  print(f"100 partials[0].time_range()={partials[0].time_range()}") # qwe

  # resp = lambda f:1.0 # no filtering
  # resp = lambda f:instruments.log_comb_response(f)
  resp = lambda f:instruments.fisher_response(f)
  for partial in partials:
    partial.filter(resp)

  print(f"200 partials[0].time_range()={partials[0].time_range()}") # qwe
  osc = Oscillator({'n_samples':n_samples,'n_instances':n_instances,'t0':0.0,'dt':1/sample_freq},partials)

  if False:
    print("graphing...")
    partials[10].a.graph("a.png",0,3,100)
    print("...done")

  timer_start = time.perf_counter()
  osc.run(dev,local_size)
  timer_end = time.perf_counter()
   
  print("return code=",osc.error_code())
  if osc.error_code()!=0:
    sys.exit(" ******* exiting with an error **********")
  print("wall-lock time for computation = ",(timer_end-timer_start)*1000,"ms")

  write_file('a.wav',osc.y(),n_samples,sample_freq)

def write_file(filename,y,n_samples,sample_freq):
  max_abs = 0.0
  illegal_at = -1
  illegal_value = 0
  for i in range(n_samples):
    if math.isnan(y[i]) or y[i]<-32767.0 or y[i]>32767.0:
      if illegal_at== -1:
        illegal_at = i
        illegal_value = y[i]
      y[i] = 0
  if illegal_at>0:
    print("warning, illegal values in output data, first is at i=",illegal_at,", value=",illegal_value)
  for i in range(n_samples):
    if abs(y[i]>max_abs):
      max_abs = y[i]
  # Write to a file.
  # convert to 16-bit signed for WAV or AIFF
  if max_abs>0.0:
    gain = 32760.0/max_abs
  else:
    gain = 1.0
  gain *= 0.3
  pcm = numpy.zeros(n_samples, numpy.int16)
  for i in range(n_samples):
    pcm[i] = gain*y[i]
  f = wave.open(filename,'w')
  f.setnchannels(1) # mono
  f.setsampwidth(2) # 16 bits
  f.setframerate(sample_freq)
  f.writeframesraw(pcm)
  f.close()

def die(message):
  sys.exit(message)

main()

