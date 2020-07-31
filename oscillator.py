import numpy,functools,scipy,math,ctypes,sys,functools
from scipy import interpolate
import pyopencl as cl


class Oscillator:
  cpu_c_lib = ctypes.cdll.LoadLibrary('./cpu_c.so')
  MAX_SPLINE_KNOTS,SPLINE_ORDER,MAX_SPLINE_COEFFS,MAX_PARTIALS,MAX_INSTANCES = (
               cpu_c_lib.get_max_sizes(0),cpu_c_lib.get_max_sizes(1),cpu_c_lib.get_max_sizes(2),cpu_c_lib.get_max_sizes(3),cpu_c_lib.get_max_sizes(4))
  def __init__(self,pars):
    self.os = [OscillatorLowLevel(self,pars)]

  def error_code(self):
    return self.os[0].error_code()

  def setup(self,partials):
    return self.os[0].setup(partials)

  def __str__(self):
    return str(self.os[0])

  def run(self,dev,n_instances,local_size):
    self.os[0].run(dev,n_instances,local_size)

  def y(self): # results of synthesis
    return self.os[0].y

class OscillatorLowLevel:
  def __init__(self,parent,pars):
    self.parent = parent
    self.n_samples,self.samples_per_instance,self.t0,self.dt = (pars['n_samples'],pars['samples_per_instance'],pars['t0'],pars['dt'])
    # buffer to hold synthesized sound:
    self.y = numpy.zeros(self.n_samples, numpy.float32)
    # misc data structures:
    self.clear_small_arrays()

  def error_code(self):
    return self.err[0]

  def clear(self):
    self.y.fill(0.0)
    self.clear_small_arrays()

  def clear_small_arrays(self):
    self.err = numpy.zeros(Oscillator.MAX_INSTANCES, numpy.int32)
    self.error_details = numpy.zeros(Oscillator.MAX_INSTANCES*64, numpy.int32)
    self.info = numpy.zeros(100, numpy.float32)
    self.n_info = numpy.zeros(1, numpy.int32)
    self.phi_c = numpy.zeros(Oscillator.MAX_SPLINE_COEFFS, numpy.float32)
    self.phi_knots = numpy.zeros(Oscillator.MAX_SPLINE_KNOTS, numpy.float32)
    self.a_c = numpy.zeros(Oscillator.MAX_SPLINE_COEFFS, numpy.float32)
    self.a_knots = numpy.zeros(Oscillator.MAX_SPLINE_KNOTS, numpy.float32)
    self.phi_n = numpy.zeros(Oscillator.MAX_PARTIALS, numpy.int32)
    self.a_n = numpy.zeros(Oscillator.MAX_PARTIALS, numpy.int32)
    self.i_pars = numpy.zeros(100, numpy.int64)
    self.f_pars = numpy.zeros(100, numpy.float32)

  def setup(self,partials):
    self.clear()
    self.partials = partials
    n_phi_knots = self.count_knots("phi")
    n_a_knots = self.count_knots("a")
    if n_phi_knots>Oscillator.MAX_SPLINE_KNOTS or n_a_knots>Oscillator.MAX_SPLINE_KNOTS:
      raise Exception(f"too many knots, {n_phi_knots} phi or {n_a_knots} A greater than {Oscillator.MAX_SPLINE_KNOTS}")
    # create flattened versions of input data for consumption by opencl
    two_pi = 2.0*math.pi
    copy_into_numpy_array(self.phi_knots,     functools.reduce(cat,list(map(lambda p:p.phi.x,partials))) )
    copy_into_numpy_array(self.a_knots,       functools.reduce(cat,list(map(lambda p:p.a.x,    partials))) )
    copy_into_numpy_array(self.phi_c,         functools.reduce(cat,list(map(lambda p:p.phi.c.flatten(),partials))) )
    copy_into_numpy_array(self.a_c,           functools.reduce(cat,list(map(lambda p:p.a.c.flatten(),    partials))) )
    for i in range(len(partials)):
      p = partials[i]
      self.phi_n[i] = len(p.phi.x)
      self.a_n[i] = len(p.a.x)
    t1 = self.t0+self.dt*self.n_samples
    if not (self.in_time_range(self.t0) and self.in_time_range(t1)):
      raise Exception("illegal time range, t={self.t0} to {t1}, range={self.time_range()}")
    self.f_pars[0] = self.t0
    self.f_pars[1] = self.dt
    self.i_pars[0] = self.samples_per_instance
    self.i_pars[1] = len(partials)
    self.i_pars[2] = self.n_samples

  def count_knots(self,which):
    if which=="phi":
      l = lambda p:p.phi.x
    else:
      l = lambda p:p.a.x
    return len(functools.reduce(cat,list(map(l,self.partials))))

  def time_range(self):
    a,b = self.partials[0].time_range()
    for p in self.partials:
      (aa,bb) = p.time_range()
      a = max(a,aa)
      b = min(b,bb)
    return (a,b)    

  def in_time_range(self,t):
    a,b = self.time_range()
    return (t>=a and t<=b)

  def __str__(self):
    result = ''
    result = result + "valid time range = "+str(self.time_range())+"\n"
    result = result + "a_knots = "+sa(self.a_knots)+"\n"
    result = result + "a_c = "+sa(self.a_c)+"\n"
    return result

  def run(self,dev,n_instances,local_size):
    if n_instances%local_size!=0:
      raise Exception(f"local_size={local_size} is not a divisor of n_instances={n_instances}")
    if n_instances>self.parent.MAX_INSTANCES:
      raise Exception(f"n_instances={n_instances} is greater than {self.parent.MAX_INSTANCES}")

    mem_flags = cl.mem_flags
    context = dev.context
    program = dev.program
    queue = dev.queue

    y_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, self.y.nbytes) # This doesn't initialize it to 0, because write only.
    err_buf = cl.Buffer(context, mem_flags.WRITE_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.err)
    # Each instance gets its own 32-bit ints for error reporting. These are write-only, which limits what the kernel can do.
    # Each kernel starts by writing 0 to its flag. If there's an error, it overwrites that with a code that packs some error info in it.
    error_details_buf  = cl.Buffer(context, mem_flags.WRITE_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.error_details)
    info_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.info)
    n_info_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.n_info)
    phi_c_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.phi_c)
    phi_knots_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.phi_knots)
    a_c_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.a_c)
    a_knots_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.a_knots)
    phi_n_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.phi_n)
    a_n_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.a_n)
    i_pars_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.i_pars)
    f_pars_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.f_pars)

    # Estimate total size of all global arrays that will be copied to local memory:
    to_local = [phi_n_buf,a_n_buf,phi_knots_buf,phi_c_buf,a_c_buf]
    tot_local_per_instance = functools.reduce(lambda a,b:a+b,list(map(lambda x:sys.getsizeof(x),to_local)))
    workgroup_size = n_instances/local_size
    print(f"estimated total size of local arrays: {tot_local_per_instance*workgroup_size/1024.0} kb per workgroup")
   
    program.oscillator(queue, (n_instances,), (local_size,),
                       y_buf,
                       err_buf,error_details_buf,info_buf,n_info_buf,
                       phi_c_buf, phi_knots_buf, a_c_buf, a_knots_buf, phi_n_buf, a_n_buf,
                       i_pars_buf,f_pars_buf)
    # cf. clEnqueueNDRangeKernel , enqueue_nd_range_kernel 
    # This seems to be calling the __call__ method of a Kernel object, https://documen.tician.de/pyopencl/runtime_program.html
    # Args are (queue,global_size,local_size,*args).
    # global_size is size of m-dim rectangular grid, one work item launched for each point
    # local_size is size of workgroup, must be an integer divisor of global_size

    cl.enqueue_copy(queue, self.err, err_buf)
    cl.enqueue_copy(queue, self.error_details, error_details_buf)
    cl.enqueue_copy(queue, self.y, y_buf)

    have_errors = False
    for i in range(n_instances):
      if self.err[i]!=0:
        print(f"instance {i}, error={error_to_string(int(self.err[i]/1000))}, oscillator.cl line {self.err[i]%1000}")
        k = 64*i
        print(f"  details: {self.error_details[k]}, {self.error_details[k+1]}, {self.error_details[k+2]}")
        have_errors = True
    if have_errors:
      raise Exception("dying with errors")

def error_to_string(n):
  return {1:"undefined function",2:"spline too large",3:"too many partials",4:"too many knots in spline",
          5:"unexpected NaN",6:"index out of range",7:"illegal value"}[n]
  # ... defined in constants.h

def sa(a):
  # make an array into a string, omitting trailing zeroes
  last_nonzero = 0
  for i in range(len(a)):
    if a[i]!=0:
      last_nonzero = i
  return str(a[0:last_nonzero+1])

def copy_into_numpy_array(x,y):
  for i in range(len(y)):
    x[i] = y[i]

# concatenate two lists
def cat(l1,l2):
  return [*l1,*l2]
