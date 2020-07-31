#!/bin/python3

import wave,time,math,sys,copy
import pyopencl as cl
from pyopencl import array
import numpy,scipy

from opencl_device import OpenClDevice
from oscillator import Oscillator
from partial import Partial
from pie import Pie
from scipy.interpolate import CubicSpline # qwe
import vibrato

def main():
  dev = OpenClDevice()
  dev.build('oscillator.cl')
   
  length_sec = 3.0
  if False:
    n_instances = 1024 
    local_size = 64 # must divide n_instances
  else:
    n_instances = 1024
    local_size = 64
  sample_freq = 44100.0
  samples_per_instance = int(length_sec*sample_freq/n_instances)
  n_samples = n_instances*samples_per_instance

  osc = Oscillator({'n_samples':n_samples,'samples_per_instance':samples_per_instance,'t0':0.0,'dt':1/sample_freq})

  vib = vibrato.generate(100,3,3,6,0.4,[3,5,4,1],[1,8,4,1])
  p1 = Partial(vib,Pie.from_string("0 0,0.2 0.5 c ; , 2 1 ; , 3 0"))
  p2 = p1.scale_f(2).scale_a(1/math.sqrt(0.5))
  p3 = p1.scale_f(3).scale_a(1/math.sqrt(0.3))
  p4 = p1.scale_f(4).scale_a(1/math.sqrt(0.5))
  p5 = p1.scale_f(5).scale_a(1/math.sqrt(0.35))
  p6 = p1.scale_f(6).scale_a(1/math.sqrt(0.25))
  p7 = p1.scale_f(7).scale_a(1/math.sqrt(0.25))
  p8 = p1.scale_f(8).scale_a(1/math.sqrt(0.1))

  osc.setup([ p1,p2,p3,p4,p5,p6,p7,p8  ])

  p1.f.graph("a.png",0,4,100) # make a graph of the frequency of the fundamental

  timer_start = time.perf_counter()
  osc.run(dev,n_instances,local_size)
  timer_end = time.perf_counter()
   
  print("return code=",osc.error_code())
  if osc.error_code()!=0:
    sys.exit(" ******* exiting with an error **********")
  print("wall-lock time for computation = ",(timer_end-timer_start)*1000,"ms")
  y = osc.y()
  print(y)

  write_file('a.wav',y,n_samples,sample_freq)

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

