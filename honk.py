#!/bin/python3

import pyopencl as cl
from pyopencl import array
import numpy

# based on code from   https://www.drdobbs.com/open-source/easy-opencl-with-python/240162614
 
def main():
  n = 16
  a = numpy.zeros(n, numpy.float32)
   
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
   
  mem_flags = cl.mem_flags
  a_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=a)
  y = numpy.zeros(n, numpy.float32)
  y_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, y.nbytes)
   
  program.oscillator(queue, (n,), (2,), y_buf, a_buf)
  # cf. clEnqueueNDRangeKernel , enqueue_nd_range_kernel 
  # This seems to be calling the __call__ method of a Kernel object, https://documen.tician.de/pyopencl/runtime_program.html
  # Args are (queue,global_size,local_size,*args).
  # global_size is size of m-dim rectangular grid, one work item launched for each point
  # local_size is size of workgroup, must be an integer divisor of global_size
   
  cl.enqueue_copy(queue, y, y_buf)
   
  print(y)

main()
