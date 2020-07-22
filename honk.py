#!/bin/python3

import pyopencl as cl
from pyopencl import array
import numpy

# based on code from   https://www.drdobbs.com/open-source/easy-opencl-with-python/240162614
 
def main():
  n = 1024
  a = numpy.zeros(n, numpy.float32)
   
  print("number of platforms = ",len(cl.get_platforms()))

  platform = cl.get_platforms()[0]
   
  ## It would be necessary to add some code to check the check the support for
  ## the necessary platform extensions with platform.extensions
   
  device = platform.get_devices()[0]
   
  ## It would be necessary to add some code to check the check the support for
  ## the necessary device extensions with device.extensions
   
  context = cl.Context([device])
   
  with open('oscillator.cl', 'r') as f:
    opencl_code = f.read()
  program = cl.Program(context,opencl_code).build()
  #    https://documen.tician.de/pyopencl/runtime_program.html
  #    build() has optional args about caching compiled code
  #    "returns self"
   
  queue = cl.CommandQueue(context)
   
  fn = numpy.uint32(1) # zeta function
  y = numpy.zeros(n, numpy.float32)
  err = numpy.zeros(1, numpy.int32)
  info = numpy.zeros(100, numpy.float32)
  n_info = numpy.zeros(1, numpy.int32)
  v1 = numpy.zeros(100, numpy.float32)
  v2 = numpy.zeros(100, numpy.float32)
  v3 = numpy.zeros(100, numpy.float32)
  v4 = numpy.zeros(100, numpy.float32)
  i_pars = numpy.zeros(100, numpy.int64)
  f_pars = numpy.zeros(100, numpy.float32)

  mem_flags = cl.mem_flags
  y_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, y.nbytes)
  err_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=err)
  info_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=info)
  n_info_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=n_info)
  v1_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=v1)
  v2_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=v2)
  v3_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=v3)
  v4_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=v4)
  i_pars_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=i_pars)
  f_pars_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=f_pars)
   
  program.oscillator(queue, (n,), (64,),
                     fn, y_buf,
                     err_buf,info_buf,n_info_buf,
                     v1_buf, v2_buf, v3_buf, v4_buf,
                     i_pars_buf,f_pars_buf)
  # cf. clEnqueueNDRangeKernel , enqueue_nd_range_kernel 
  # This seems to be calling the __call__ method of a Kernel object, https://documen.tician.de/pyopencl/runtime_program.html
  # Args are (queue,global_size,local_size,*args).
  # global_size is size of m-dim rectangular grid, one work item launched for each point
  # local_size is size of workgroup, must be an integer divisor of global_size
   
  cl.enqueue_copy(queue, y, y_buf)
   
  print(y)

main()
