import pyopencl as cl

class OpenClDevice:
  def __init__(self):
    self.platform = cl.get_platforms()[0]
    self.device = self.platform.get_devices()[0]
    self.context = cl.Context([self.device])

  def build(self,source_filename):
    with open(source_filename, 'r') as f:
      opencl_code = f.read()
    self.program = cl.Program(self.context,opencl_code).build()
    #    https://documen.tician.de/pyopencl/runtime_program.html
    #    build() has optional args about caching compiled code
    #    "returns self"
    self.queue = cl.CommandQueue(self.context) # does this need to be after we compile?

