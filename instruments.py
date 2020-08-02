import math,numpy,scipy

def violin_envelope(n_partials=100,instrument="violin",f=None,brightness=0,what="bd",norm=1):
  """
  Return the amplitude spectrum (magnitudes only).
  Schoonderwaldt, 2009, "The Violinist’s Sound Palette: Spectral Centroid, Pitch Flattening and Anomalous Low Frequencies."
  n_partials= number of partials to return
  brightness=0 for ordinario or mf, 1=sul ponticello or f, -1=sul tasto or p. This controlls the roll-off frequency described by Schoonderwaldt.
  what="bd" for displacement at the bridge
  f=frequency
  norm -- square root of sum of squares of amplitudes is normalized to this value
  """
  roll_off_octave = 4.2+0.4*brightness # octaves above the fundamental at which the roll-off occurs
  roll_off = 2.0**roll_off_octave # frequency in units of fundamental
  bd = [] # amplitudes of bridge displacement
  for i in range(n_partials):
    n=i+1 # 1=fundamental
    a = n**-2
    # ... is the envelope of the fourier series of a triangle wave, neglecting the oscillations in the sinc function, which are not
    #     actually observed at the bridge; in Schoonderwaldt's experimental spectra, which I assume are of |a|^2, this shows up as
    #     6 dB per octave
    r = n-roll_off # number of octaves by which we exceed the roll-off (arrows in Schoonderwaldt's fig. 5)
    roll_factor = 1.0
    if r>0:
      knee = 0.3 # width of the knee in octaves
      max_db = 10.0 # power, which I think is what's shown in Schoonderwaldt's fig. 5
      if r>knee:
        roll_off_db = max_db
      else:
        roll_off_db = (r/knee)*max_db
      roll_factor = 10.0**(-0.5*roll_off_db/10.0) # 0.5 because I'm computing amplitude, not power
    a = a*roll_factor
    bd.append(a)
  if what=="bd":
    return normalize(bd,norm)
  raise Exception(f"what={what} not recognized")

def normalize(amplitudes,norm):
  s = 0.0
  for a in amplitudes:
    s = s+a*a
  s = math.sqrt(s)
  k = norm/s
  return k*numpy.array(amplitudes)

def log_comb_response(f,contrast=10,spacing=1.2):
  """
  Simulate a response function of the type described by Mathews and Kohut, 1973, Electronic simulation of violin resonances.
  Spacing of filter is in units of whole-tones. Mathews lists frequencies that are about 1.4 whole-steps apart at low frequencies,
  more like 1.0 at the high ones. Contrast is dB power.
  Output is a gain (in linear amplitude units).
  """
  x = 54.388*math.log(f)/spacing # the numerical factor is (6/ln2)(2pi)
  y = 0.057565*contrast*sin(x)
  # ... the numerical factor is (1/2)(1/2)(1/10)ln 10; reasons for factors are as follows:
  #     1/2 ... sine function has a peak-to-peak variation of 2 units
  #     1/2 ... put it in linear amplitude units (as opposed to contrast, which is in db power units)
  #     1/10 ... because it's *deci*bels
  #     ln 10 ... base 10
  return math.exp(y)
