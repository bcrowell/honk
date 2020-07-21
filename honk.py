#!/bin/python3

import pyopencl as cl
from pyopencl import array
import numpy
 
def main():
  n = 16
  a = numpy.zeros(n, numpy.float32)
   
  ## Step #1. Obtain an OpenCL platform.
  platform = cl.get_platforms()[0]
   
  ## It would be necessary to add some code to check the check the support for
  ## the necessary platform extensions with platform.extensions
   
  ## Step #2. Obtain a device id for at least one device (accelerator).
  device = platform.get_devices()[0]
   
  ## It would be necessary to add some code to check the check the support for
  ## the necessary device extensions with device.extensions
   
  ## Step #3. Create a context for the selected device.
  context = cl.Context([device])
   
  ## Step #4. Create the accelerator program from source code.
  ## Step #5. Build the program.
  ## Step #6. Create one or more kernels from the program functions.
  with open('oscillator.cl', 'r') as f:
    opencl_code = f.read()
  program = cl.Program(context,opencl_code).build()
   
  ## Step #7. Create a command queue for the target device.
  queue = cl.CommandQueue(context)
   
  ## Step #8. Allocate device memory and move input data from the host to the device memory.
  mem_flags = cl.mem_flags
  a_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=a)
  y = numpy.zeros(n, numpy.float32)
  y_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, y.nbytes)
   
  ## Step #9. Associate the arguments to the kernel with kernel object.
  ## Step #10. Deploy the kernel for device execution.
  program.oscillator(queue, y.shape, None, a_buf, y_buf)
   
  ## Step #11. Move the kernel’s output data to host memory.
  cl.enqueue_copy(queue, y, y_buf)
   
  ## Step #12. Release context, program, kernels and memory.
  ## PyOpenCL performs this step for you, and therefore,
  ## you don't need to worry about cleanup code
   
  print(y)

main()
