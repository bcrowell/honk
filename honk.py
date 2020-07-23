#!/bin/python3

import pyopencl as cl
from pyopencl import array
import numpy

# based on code from   https://www.drdobbs.com/open-source/easy-opencl-with-python/240162614
 
def main():
  print("number of platforms = ",len(cl.get_platforms()))
  platform = cl.get_platforms()[0]
  device = platform.get_devices()[0]
  context = cl.Context([device])
   
  with open('oscillator.cl', 'r') as f:
    opencl_code = f.read()
  program = cl.Program(context,opencl_code).build()
  #    https://documen.tician.de/pyopencl/runtime_program.html
  #    build() has optional args about caching compiled code
  #    "returns self"
   
  queue = cl.CommandQueue(context)
   
  n = 1024 # number of instances
  samples_per_instance = 50
  n_samples = n*samples_per_instance

  fn = numpy.uint32(1) # cubic spline oscillator
  y = numpy.zeros(n_samples, numpy.float32)
  err = numpy.zeros(1, numpy.int32)
  info = numpy.zeros(100, numpy.float32)
  n_info = numpy.zeros(1, numpy.int32)
  v1 = numpy.zeros(100, numpy.float32)
  v2 = numpy.zeros(100, numpy.float32)
  v3 = numpy.zeros(100, numpy.float32)
  v4 = numpy.zeros(100, numpy.float32)
  i_pars = numpy.zeros(100, numpy.int64)
  f_pars = numpy.zeros(100, numpy.float32)

  # knots
  v2[0] = 0.0; v2[1] = 1.0;
  v4[0] = 0.0; v4[1] = 1.0;
  # constant spline polynomials
  v1[3] = 1.0; v1[4] = 1.0; # constant omega=1
  v3[3] = 1.0; v3[4] = 1.0; # constant amplitude=1
  # scalar parameters:
  i_pars[0] = 2; # n for omega spline
  i_pars[1] = 2; # n for amplitude spline
  i_pars[2] = samples_per_instance;
  f_pars[0] = 0.0; # phase
  f_pars[1] = 0.0; # t0
  f_pars[2] = 0.001; # dt

  mem_flags = cl.mem_flags
  fn_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=fn)
  y_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, y.nbytes)
  err_buf = cl.Buffer(context, mem_flags.WRITE_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=err)
  info_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=info)
  n_info_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=n_info)
  v1_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=v1)
  v2_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=v2)
  v3_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=v3)
  v4_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=v4)
  i_pars_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=i_pars)
  f_pars_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=f_pars)
   
  program.oscillator(queue, (n,), (64,),
                     fn_buf, y_buf,
                     err_buf,info_buf,n_info_buf,
                     v1_buf, v2_buf, v3_buf, v4_buf,
                     i_pars_buf,f_pars_buf)
  # cf. clEnqueueNDRangeKernel , enqueue_nd_range_kernel 
  # This seems to be calling the __call__ method of a Kernel object, https://documen.tician.de/pyopencl/runtime_program.html
  # Args are (queue,global_size,local_size,*args).
  # global_size is size of m-dim rectangular grid, one work item launched for each point
  # local_size is size of workgroup, must be an integer divisor of global_size
   
  cl.enqueue_copy(queue, err, err_buf)
  cl.enqueue_copy(queue, y, y_buf)
   
  print("return code=",err)
  print(y)

main()
