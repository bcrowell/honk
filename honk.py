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
   
  queue = cl.CommandQueue(context)
   
  mem_flags = cl.mem_flags
  a_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=a)
  y = numpy.zeros(n, numpy.float32)
  y_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, y.nbytes)
   
  program.oscillator(queue, y.shape, None, a_buf, y_buf)
   
  cl.enqueue_copy(queue, y, y_buf)
   
  print(y)

main()
