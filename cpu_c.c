/*
  C code meant to run on the CPU rather than the GPU, and to be called
  from python.
*/

#include "constants.h"

int get_max_sizes(int what) {
  if (what==0) {return MAX_SPLINE_KNOTS;}
  if (what==1) {return SPLINE_ORDER;}
  if (what==2) {return MAX_SPLINE_COEFFS;}
  if (what==3) {return MAX_PARTIALS;}
  return -1;
}

