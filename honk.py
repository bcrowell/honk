#!/bin/python3

import wave,time,math,sys,ctypes
import pyopencl as cl
from pyopencl import array
import numpy

from opencl_device import OpenClDevice
from oscillator import Oscillator
from partial import Partial

# based on code from   https://www.drdobbs.com/open-source/easy-opencl-with-python/240162614
 
def main():
  cpu_c_lib = ctypes.cdll.LoadLibrary('./cpu_c.so')
  max_spline_knots,spline_order,max_spline_coeffs,max_partials = (cpu_c_lib.get_max_sizes(0),cpu_c_lib.get_max_sizes(1),
                                                                  cpu_c_lib.get_max_sizes(2),cpu_c_lib.get_max_sizes(3))

  dev = OpenClDevice()
  dev.build('oscillator.cl')
   
  n_instances = 1024
  sample_freq = 44100.0
  length_sec = 1.0
  samples_per_instance = int(length_sec*sample_freq/n_instances)
  n_samples = n_instances*samples_per_instance

  osc = Oscillator(n_samples,max_spline_knots,spline_order,max_spline_coeffs,max_partials)

  p1 = Partial([0,1],[1000,1000],[0,1],[1,1],0)
  p2 = Partial([0,1],[2000,2000],[0,1],[1,1],0)


  osc.setup([p1,p2])

  osc.i_pars[0] = samples_per_instance;
  osc.f_pars[0] = 0.0; # t0
  osc.f_pars[1] = 1/sample_freq; # dt

  timer_start = time.perf_counter()
  do_oscillator(osc.y,dev,n_instances,
                osc.err,osc.info,osc.n_info,osc.omega_c,osc.omega_knots,osc.a_c,osc.a_knots,osc.phase,osc.omega_n,osc.a_n,osc.i_pars,osc.f_pars)
  timer_end = time.perf_counter()
   
  print("return code=",osc.error_code())
  if osc.error_code()!=0:
    sys.exit(" ******* exiting with an error **********")
  print("time = ",(timer_end-timer_start)*1000,"ms")
  print(osc.y)

  write_file('a.wav',osc.y,n_samples,sample_freq)

def write_file(filename,y,n_samples,sample_freq):
  max_abs = 0.0
  for i in range(n_samples):
    if abs(y[i]>max_abs):
      max_abs = y[i]
  # Write to a file.
  # convert to 16-bit signed for WAV or AIFF
  if max_abs>0.0:
    gain = 32760.0/max_abs
  else:
    gain = 1.0
  pcm = numpy.zeros(n_samples, numpy.int16)
  for i in range(n_samples):
    pcm[i] = gain*y[i]
  f = wave.open(filename,'w')
  f.setnchannels(1) # mono
  f.setsampwidth(2) # 16 bits
  f.setframerate(sample_freq)
  f.writeframesraw(pcm)
  f.close()

def do_oscillator(y,dev,n_instances,err,info,n_info,omega_c,omega_knots,a_c,a_knots,phase,omega_n,a_n,i_pars,f_pars):
  mem_flags = cl.mem_flags
  context = dev.context
  program = dev.program
  queue = dev.queue
  y_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, y.nbytes)
  err_buf = cl.Buffer(context, mem_flags.WRITE_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=err)
  info_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=info)
  n_info_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=n_info)
  omega_c_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=omega_c)
  omega_knots_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=omega_knots)
  a_c_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=a_c)
  a_knots_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=a_knots)
  phase_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=phase)
  omega_n_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=omega_n)
  a_n_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=a_n)
  i_pars_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=i_pars)
  f_pars_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=f_pars)
   
  program.oscillator(queue, (n_instances,), (64,),
                     y_buf,
                     err_buf,info_buf,n_info_buf,
                     omega_c_buf, omega_knots_buf, a_c_buf, a_knots_buf, phase_buf, omega_n_buf, a_n_buf,
                     i_pars_buf,f_pars_buf)
  # cf. clEnqueueNDRangeKernel , enqueue_nd_range_kernel 
  # This seems to be calling the __call__ method of a Kernel object, https://documen.tician.de/pyopencl/runtime_program.html
  # Args are (queue,global_size,local_size,*args).
  # global_size is size of m-dim rectangular grid, one work item launched for each point
  # local_size is size of workgroup, must be an integer divisor of global_size
   
  cl.enqueue_copy(queue, err, err_buf)
  cl.enqueue_copy(queue, y, y_buf)

main()

