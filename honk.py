#!/bin/python3

import wave,time,math,sys,ctypes,copy
import pyopencl as cl
from pyopencl import array
import numpy,scipy

from opencl_device import OpenClDevice
from oscillator import Oscillator
from partial import Partial
from pie import Pie

# based on code from   https://www.drdobbs.com/open-source/easy-opencl-with-python/240162614
 
def main():
  cpu_c_lib = ctypes.cdll.LoadLibrary('./cpu_c.so')
  max_spline_knots,spline_order,max_spline_coeffs,max_partials = (cpu_c_lib.get_max_sizes(0),cpu_c_lib.get_max_sizes(1),
                                                                  cpu_c_lib.get_max_sizes(2),cpu_c_lib.get_max_sizes(3))

  dev = OpenClDevice()
  dev.build('oscillator.cl')
   
  length_sec = 4.0
  n_instances = 1024
  sample_freq = 44100.0
  samples_per_instance = int(length_sec*sample_freq/n_instances)
  n_samples = n_instances*samples_per_instance

  osc = Oscillator(n_samples,max_spline_knots,spline_order,max_spline_coeffs,max_partials)

  osc.setup([
    Partial(
              Pie(scipy.interpolate.CubicSpline([0.0,2.0],[200.0,200.0])).extend(
              Pie(scipy.interpolate.CubicSpline([2.0,4.0],[224.0,224.0])))
            ,
            Pie(scipy.interpolate.CubicSpline([0.0,4.0],[1,1])),
            0)
  ])

  osc.i_pars[0] = samples_per_instance;
  osc.f_pars[0] = 0.0; # t0
  osc.f_pars[1] = 1/sample_freq; # dt

  #print(osc)

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
  omega_c_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.omega_c)
  omega_knots_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.omega_knots)
  a_c_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.a_c)
  a_knots_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.a_knots)
  phase_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.phase)
  omega_n_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=osc.omega_n)
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
                     omega_c_buf, omega_knots_buf, a_c_buf, a_knots_buf, phase_buf, omega_n_buf, a_n_buf,
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

