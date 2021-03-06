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
  vib = vibrato.generate(290,3,3,6,0.4,[3,5,4,1],[1,8,4,1])
  partials = []
  for i in range(len(a)):
    n=i+1 # n=1 for fundamental
    p = Partial(vib,Pie.from_string("0 0,0.2 0.5 c ; , 2 1 ; , 3 0")) # gradual onset
    p = copy.deepcopy(p).scale_f(n).scale_a(a[i])
    partials.append(p)

  # resp = lambda f:1.0 # no filtering
  # resp = lambda f:instruments.log_comb_response(f)
  resp = lambda f:instruments.fisher_response(f)
  for partial in partials:
    partial.filter(resp)

  osc = Oscillator({'n_samples':n_samples,'n_instances':n_instances,'t0':0.0,'dt':1/sample_freq},partials)

  if False:
    print("graphing...")
    #attack_envelope.graph("a.png",0,3,100)
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

