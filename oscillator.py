import numpy,functools,scipy,math,ctypes,sys,functools,copy
from scipy import interpolate
import pyopencl as cl


class Oscillator:
  """
  A list of OscillatorSeq objects that are simultaneous and need to be added at the end.
  """
  cpu_c_lib = ctypes.cdll.LoadLibrary('./cpu_c.so')
  MAX_SPLINE_KNOTS,SPLINE_ORDER,MAX_SPLINE_COEFFS,MAX_PARTIALS,MAX_INSTANCES = (
               cpu_c_lib.get_max_sizes(0),cpu_c_lib.get_max_sizes(1),cpu_c_lib.get_max_sizes(2),cpu_c_lib.get_max_sizes(3),cpu_c_lib.get_max_sizes(4))
  def __init__(self,pars,partials):
    # pars should contain keys n_samples, n_instances, t0, and dt
    maxp = Oscillator.MAX_PARTIALS
    n_sets = int(len(partials)/maxp)
    if n_sets*maxp<len(partials):
      n_sets += 1
    self.oseqs = []
    for j in range(n_sets):
      this_set = []
      k1 = j*maxp
      k2 = (j+1)*maxp-1
      if k2>len(partials)-1:
        k2=len(partials)-1
      for k in range(k1,k2+1): # range doesn't include upper arg, so add 1
        this_set.append(partials[k])
      self.oseqs.append(OscillatorSeq(pars,this_set))

  def error_code(self):
    for o in self.oseqs:
      if o.error_code()!=0:
        return o.error_code()
    return 0

  def run(self,dev,local_size):
    for o in self.oseqs:
      o.run(dev,local_size)

  def y(self): # results of synthesis
    yy = None
    first_one = True
    for o in self.oseqs:
      this_y = o.y()
      if first_one:
        yy = this_y
        first_one = False
      else:
        yy = numpy.add(yy,this_y)
    return yy

class OscillatorSeq:
  """
  A list of OscillatorLowLevel objects that occur sequentially in time.
  """
  cpu_c_lib = ctypes.cdll.LoadLibrary('./cpu_c.so')
  MAX_SPLINE_KNOTS,SPLINE_ORDER,MAX_SPLINE_COEFFS,MAX_PARTIALS,MAX_INSTANCES = (
               cpu_c_lib.get_max_sizes(0),cpu_c_lib.get_max_sizes(1),cpu_c_lib.get_max_sizes(2),cpu_c_lib.get_max_sizes(3),cpu_c_lib.get_max_sizes(4))
  def __init__(self,pars,partials):
    # pars should contain keys n_samples, n_instances, t0, and dt
    self.n_samples,self.t0,self.dt,self.n_instances = (
          pars['n_samples'],pars['t0'],pars['dt'],pars['n_instances'])
    if self.too_big_horizontally(partials):
      n_samples,t0,dt = (pars['n_samples'],pars['t0'],pars['dt'])
      if n_samples==0:
        raise Exception("recursion failed to bottom out")
      nn = [0,0]
      nn[0] = int(n_samples/2)
      nn[1] = n_samples-nn[0]
      subs = []
      for i in range(2):
        sub_pars = copy.deepcopy(pars)
        sub_pars['n_samples'] = nn[i]
        if i==0:
          sub_t0 = t0
        else:
          sub_t0 = t0+nn[0]*dt
        sub_t1 = sub_t0+(nn[i]-1)*dt
        sub_pars['t0'] = sub_t0
        sub_partials = []
        for p in partials:
          sub_partials.append(p.restrict(sub_t0,sub_t1))
        subs.append(OscillatorSeq(sub_pars,sub_partials))
      self.os = subs[0].os
      self.cat(subs[1])
      return
    # If we get to here, we didn't need to recurse.
    n_samples = pars['n_samples']
    n_instances = pars['n_instances']
    samples_per_instance = int(n_samples/n_instances)
    if samples_per_instance*n_instances<n_samples:
      samples_per_instance += 1
    pars2 = copy.deepcopy(pars)
    pars2['samples_per_instance'] = samples_per_instance
    self.os = [OscillatorLowLevel(self,pars2)]
    for o in self.os:
      o.setup(partials)

  def cat(self,osc):
    # concatenate two oscillators; doesn't return anything
    self.os.extend(osc.os)

  def error_code(self):
    for o in self.os:
      if o.error_code()!=0:
        return o.error_code()
    return 0

  def __str__(self):
    s = 'Oscillator:\n'
    for o in self.os:
      s = s + str(o)
    return s

  def run(self,dev,local_size):
    for o in self.os:
      o.run(dev,o.n_instances,local_size)

  def y(self): # results of synthesis
    yy = []
    for o in self.os:
      yy.extend(o.y)
    return yy

  def too_big_horizontally(self,partials):
    n_phi_knots = self.count_knots(partials,"phi")
    n_a_knots = self.count_knots(partials,"a")
    if False:
      print("artificially causing a horizontal split")
      return (n_phi_knots>200 or n_a_knots>200)
    return (n_phi_knots>Oscillator.MAX_SPLINE_KNOTS or n_a_knots>Oscillator.MAX_SPLINE_KNOTS)

  def count_knots(self,partials,which):
    if which=="phi":
      l = lambda p:p.phi.x
    else:
      l = lambda p:p.a.x
    return len(functools.reduce(cat,list(map(l,partials))))

class OscillatorLowLevel:
  def __init__(self,parent,pars):
    # pars should contain keys n_samples, samples_per_instance, n_instances, t0, and dt
    self.parent = parent
    self.n_samples,self.samples_per_instance,self.t0,self.dt,self.n_instances = (
          pars['n_samples'],pars['samples_per_instance'],pars['t0'],pars['dt'],pars['n_instances'])
    # buffer to hold synthesized sound:
    self.y = numpy.zeros(self.n_samples, numpy.float32)
    # misc data structures:
    self.clear_small_arrays()
    self.my_errors = 0

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
      raise Exception(f"illegal time range, t={self.t0} to {t1} is not within time range of partials, which is {self.defined_time_range()}")
    self.f_pars[0] = self.t0
    self.f_pars[1] = self.dt
    self.i_pars[0] = self.samples_per_instance
    self.i_pars[1] = len(partials)
    self.i_pars[2] = self.n_samples

  def defined_time_range(self):
    # intersection of domains of all partials
    a,b = self.partials[0].time_range()
    for p in self.partials:
      (aa,bb) = p.time_range()
      #print(f"from {a},{b} to {max(a,aa)},{min(b,bb)}") # qwe
      a = max(a,aa)
      b = min(b,bb)
    return (a,b)    

  def in_time_range(self,t,eps=0.0001):
    a,b = self.defined_time_range()
    return (t>=a-eps and t<=b+eps)

  def __str__(self):
    result = ''
    result = result + "defined time range = "+str(self.defined_time_range())+"\n"
    result = result + "a_knots = "+sa(self.a_knots)+"\n"
    result = result + "a_c = "+sa(self.a_c)+"\n"
    result = result + "phi_knots = "+sa(self.phi_knots)+"\n"
    result = result + "phi_c = "+sa(self.phi_c)+"\n"
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
    #print(f"estimated total size of local arrays: {tot_local_per_instance*workgroup_size/1024.0} kb per workgroup")
   
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
      self.my_errors = 1

def error_to_string(n):
  s = {1:"undefined function",2:"spline too large",3:"too many partials",4:"too many knots in spline",
          5:"unexpected NaN",6:"index out of range",7:"illegal value"}
  # ... defined in constants.h
  if n in s:
    return s[n]
  else:
    return str(n)

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
