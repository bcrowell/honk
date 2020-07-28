#!/bin/python3

import wave,time,math,sys,ctypes,copy
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
  cpu_c_lib = ctypes.cdll.LoadLibrary('./cpu_c.so')
  max_spline_knots,spline_order,max_spline_coeffs,max_partials = (cpu_c_lib.get_max_sizes(0),cpu_c_lib.get_max_sizes(1),
                                                                  cpu_c_lib.get_max_sizes(2),cpu_c_lib.get_max_sizes(3))

  dev = OpenClDevice()
  dev.build('oscillator.cl')
   
  length_sec = 3.0
  n_instances = 1024
  sample_freq = 44100.0
  samples_per_instance = int(length_sec*sample_freq/n_instances)
  n_samples = n_instances*samples_per_instance

  osc = Oscillator(n_samples,max_spline_knots,spline_order,max_spline_coeffs,max_partials)

  vib = vibrato.generate(100,3,5,7,1,[3,5,2,1],[3,5,2,1])
  p1 = Partial(vib,Pie.from_string("0 0,0.2 0.5 c ; , 2 1 ; , 3 0"))
  if False:
    p1 = Partial(
              Pie.from_string("0.0 200,2.0 224 c ; , 2.3 224 , 2.5 214 , 2.65 234 , 2.8 214 , 2.95 234 , 3.1 214 , 3.25 234 , 3.4 234 , 3.65 224 , 4.0 224 c"),
              Pie.from_string("0 0,0.5 0.5 c ; , 2 1 ; , 3.5 1 ;  , 4.0 0 c")
              )
  p2 = p1.scale_f(3).scale_a(1.0/3.0)
  p3 = p1.scale_f(5).scale_a(1.0/5.0)
  p4 = p1.scale_f(7).scale_a(1.0/7.0)

  osc.setup([ p1,p2,p3,p4  ])

  osc.i_pars[0] = samples_per_instance;
  osc.f_pars[0] = 0.0; # t0
  osc.f_pars[1] = 1/sample_freq; # dt

  p1.f.graph("a.png",0,4,100) # make a graph of the frequency of the fundamental

  timer_start = time.perf_counter()
  do_oscillator(osc,dev,n_instances,n_samples)
  timer_end = time.perf_counter()
   
  print("return code=",osc.error_code())
  if osc.error_code()!=0:
    sys.exit(" ******* exiting with an error **********")
  print("time = ",(timer_end-timer_start)*1000,"ms")
  print(osc.y)

  write_file('a.wav',osc.y,n_samples,sample_freq)

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

def do_oscillator(osc,dev,n_instances,n_samples):
  mem_flags = cl.mem_flags
  context = dev.context
  program = dev.program
  queue = dev.queue
  y_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, osc.y.nbytes) # This doesn't initialize it to 0, because write only.
  err_buf = cl.Buffer(context, mem_flags.WRITE_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.err)
  info_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.info)
  n_info_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.n_info)
  phi_c_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.phi_c)
  phi_knots_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.phi_knots)
  a_c_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.a_c)
  a_knots_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.a_knots)
  phi_n_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.phi_n)
  a_n_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.a_n)
  i_pars_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.i_pars)
  f_pars_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.f_pars)

  t0 = osc.f_pars[0]
  dt = osc.f_pars[1]
  t1 = t0+dt*n_samples
  if not (osc.in_time_range(t0) and osc.in_time_range(t1)):
    print("t=",t0,t1," range=",osc.time_range())
    die("illegal time range")
   
  program.oscillator(queue, (n_instances,), (64,),
                     y_buf,
                     err_buf,info_buf,n_info_buf,
                     phi_c_buf, phi_knots_buf, a_c_buf, a_knots_buf, phi_n_buf, a_n_buf,
                     i_pars_buf,f_pars_buf)
  # cf. clEnqueueNDRangeKernel , enqueue_nd_range_kernel 
  # This seems to be calling the __call__ method of a Kernel object, https://documen.tician.de/pyopencl/runtime_program.html
  # Args are (queue,global_size,local_size,*args).
  # global_size is size of m-dim rectangular grid, one work item launched for each point
  # local_size is size of workgroup, must be an integer divisor of global_size
   
  cl.enqueue_copy(queue, osc.err, err_buf)
  cl.enqueue_copy(queue, osc.y, y_buf)

def die(message):
  sys.exit(message)

main()

