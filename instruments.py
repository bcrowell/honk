import math,numpy,scipy
import data.violin_admittance_fisher_1787.fisher as fisher
import data.violin_bissinger_radiation_gain.bissinger as bissinger

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

def fisher_response(f):
  return fisher.admittance(f)*bissinger.radiation_gain(f)*(f/1000.0)**-0.76
  # Explanation of the third factor:
  #   This factor is because empirically, without it, the tone comes out much too bright by ear.
  #   The 1000 is just so that the factor comes out to be of order unity.
  #   I determined the exponent as follows. Found a recording with a solo violin note (Perlman, A la carte) at 290 Hz. Took its spectrum
  #   in audacity, which I assume is a power spectrum. Measured roll-off of spectrum by comparing 1st and 7th partial: -5.2 dB power per octave.
  #   Compared with synthesized note, -1.5 dB power per octave. Correction is -2.3 dB amplitude per octave, which corresponds to
  #   an exponent of ((ln10)/(10ln2))(-2.3)=-0.76. After inserting this factor, checked that the spectrum's envelope seemed to match
  #   the real violin note, and it did seem to.

def log_comb_response(f,contrast=10,spacing=1.06,i=3,ic=0.7,offset=0):
  """
  Simulate a response function of the type described by Mathews and Kohut, 1973, Electronic simulation of violin resonances.
  Spacing of filter is in units of whole-tones. Mathews lists frequencies with spacings that vary somewhat irregularly.
  Contrast is dB power. A contrast of 15 produces a clear difference from the unfiltered tone, but an uneven tone.
  Output is a gain (in linear amplitude units).
  """
  mathews = [.24,.07,-.17,-.11,.02,-.07,-.03,.01,.04,.06,.01,0.0,.03,-.02,-.13,.06,.05,-.04,.07,.06,.02,.03,-.07]
  # difference, in whole tones, between best-fit linear rule and mathews's slightly irregular frequencies

  x = 54.388*math.log(f)/spacing+offset*0.628318530717959
  # ... Pitch in wholetones, times 2pi. The first numerical factor is (6/ln2)(2pi). The second one is 2pi/10, so that
  #     an offset of 10 moves us by one whole step, or approximately one comb spacing.

  irreg = i*0.1*math.sin(ic*x)
  '''
       Add some irregularity to the spacing. The 0.1 is just so I can use integer-ish values of i.
       The value of ic is somewhat arbitrary, just meant to represent the "wavelength" of the variation in Mathews' numbers.
       I can sometimes get just barely noticeable auditory effects from this irregularity, but they're subtle.
       For all I know, Mathews' irregularities were because of what electrical components they had handy.
       I get a slight audible difference between i=0 and i=3 with contrast=15, spacing=1.06, ic=0.7. The default values
       of i and ic are chosen not because this necessarily sounded better but because it was one of the few sets of params
       for which I was able to hear any audible effect.
  '''   

  y = 0.057565*contrast*math.sin(x+irreg)
  # ... the numerical factor is (1/2)(1/2)(1/10)ln 10; reasons for factors are as follows:
  #     1/2 ... sine function has a peak-to-peak variation of 2 units
  #     1/2 ... put it in linear amplitude units (as opposed to contrast, which is in db power units)
  #     1/10 ... because it's *deci*bels
  #     ln 10 ... bels are base 10

  return math.exp(y)
