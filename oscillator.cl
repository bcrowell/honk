#ifdef RUN_ON_CPU
#include <math.h>
#endif

#define DEBUG 1

#include "gpu.h"
#include "honk.h"
#include "constants.h"

#ifndef RUN_ON_CPU
__kernel void oscillator(__global FLOAT *y,
                         __global int *err, __global FLOAT *info,__global int *n_info,
                         __global const FLOAT *v1, __global const FLOAT *v2, __global const FLOAT *v3, __global const FLOAT *v4,
                         __global const int *k1,  __global const int *k2,
                         __global const long *i_pars, __global const FLOAT *f_pars
                          ) {
  int i = get_global_id(0); // index of the current element in the computational grid
  *err = 0;
  fn_osc(y,i,err,info,n_info,v1,v2,v3,v4,k1,k2,i_pars,f_pars);
}

#endif

void fn_osc(__global FLOAT *y,int i,
                         __global int *err, __global FLOAT *info,__global int *n_info,
                         __global const FLOAT *v1, __global const FLOAT *v2,
                         __global const FLOAT *v3, __global const FLOAT *v4,
                         __global const int *k1,  __global const int *k2,
                         __global const long *i_pars, __global const FLOAT *f_pars) {
  int samples_per_instance = i_pars[0];
  int n_partials = i_pars[1];
  // copy data to local memory for efficiency:
  __local int phi_n[MAX_PARTIALS]; // phi_n[m] is the number of knots in the piecewise polynomial for the mth partial's phase
  __local int a_n[MAX_PARTIALS];
  __local FLOAT phi_knots[MAX_SPLINE_KNOTS];
  __local FLOAT a_knots[MAX_SPLINE_KNOTS];
  __local FLOAT phi_c_local[MAX_SPLINE_COEFFS];
  __local FLOAT a_c_local[MAX_SPLINE_COEFFS];
  // ---- copy number of knots
  if (n_partials>MAX_PARTIALS) {*err=HONK_ERR_TOO_MANY_PARTIALS; return;}
  for (int m=0; m<n_partials; m++) {
    phi_n[m] = k1[m];
    a_n[m]   = k2[m];
    if (phi_n[m]>MAX_SPLINE_KNOTS || a_n[m]>MAX_SPLINE_KNOTS) {*err=HONK_ERR_TOO_MANY_KNOTS_IN_SPLINE; return;}
  }
  // ---- copy locations of knots
  int k_phi = 0;
  int k_a = 0;
  int qwe = 0;
  for (int m=0; m<n_partials; m++) {
    int this_phi_n = phi_n[m];
    int this_a_n = a_n[m];
    if (k_phi+this_phi_n>MAX_SPLINE_KNOTS || k_a+this_a_n>MAX_SPLINE_KNOTS) {*err=HONK_ERR_TOO_MANY_KNOTS_IN_SPLINE; return;}
    for (int j=0; j<this_phi_n; j++) {
#if DEBUG
      if (k_phi+j>MAX_SPLINE_KNOTS) {*err=1000+m;return;}
      if (k_phi+j>MAX_SPLINE_KNOTS) {*err=HONK_INDEX_OUT_OF_RANGE;return;}
#endif
      phi_knots[k_phi+j] = v2[k_phi+j];
    }
    for (int j=0; j<this_a_n; j++) {
      if (k_a==0 && j==0 && !isnan(v4[k_a+j])) {qwe=1;}
      a_knots[k_a+j]     = v4[k_a+j];
    }
    k_phi += this_phi_n;
    k_a   += this_a_n;
  }
  // ---- copy coefficients of cubic spline polynomials
  k_phi = 0;
  k_a = 0;
  for (int m=0; m<n_partials; m++) {
    int this_phi_n = phi_n[m];
    int this_a_n = a_n[m];
    int phi_size = (this_phi_n-1)*(PHASE_SPLINE_ORDER+1); // n-1 because there are no coeffs associated with rightmost knot
    int a_size =     (this_a_n-1)    *(A_SPLINE_ORDER+1);
    if (k_phi+phi_size>MAX_SPLINE_COEFFS || k_a+a_size>MAX_SPLINE_COEFFS) {*err=HONK_ERR_SPLINE_TOO_LARGE; return;}
    for (int j=0; j<phi_size; j++) {
      phi_c_local[k_phi+j] = v1[k_phi+j]; // ************ is this right now?
    }
    for (int j=0; j<a_size; j++) {
      a_c_local[k_a+j]     = v3[k_a+j];
    }
    k_phi += phi_size;
    k_a     += a_size;
  }
#if DEBUG
  if (isnan(a_knots[0])) {*err=HONK_ERR_NAN;return;}
#endif
  oscillator_cubic_spline(y,
                          phi_c_local,phi_knots,phi_n,
                          a_c_local,    a_knots    ,a_n,
                          f_pars[0],f_pars[1],i*samples_per_instance,(i+1)*samples_per_instance-1,n_partials,
                          err
                         );
}


void oscillator_cubic_spline(__global FLOAT *y,
                             __local FLOAT *phi_c,__local FLOAT *phi_knots,__local int *phi_n,
                             __local FLOAT *a_c,    __local FLOAT *a_knots,    __local int *a_n,
                             FLOAT t0,FLOAT dt,int j1,int j2,int n_partials,
                             __global int *err) {
  FLOAT phi,a,t;
  for (int j=j1; j<=j2; j++) {
    y[j] = 0.0; // y is write-only, so it can't be initialized to zero for us
        // ... fixme -- inefficient
  }
  __local FLOAT *this_phi_c = phi_c;
  __local FLOAT *this_a_c     = a_c;
  __local FLOAT *this_phi_knots = phi_knots;
  __local FLOAT *this_a_knots = a_knots;
  __local int local_err; // for efficiency, avoid accessing the global err
  for (int m=0; m<n_partials; m++) {
    int this_phi_n = phi_n[m];
    int this_a_n = a_n[m];
    int phi_i = 0; // current knot number in spline for phi
    int a_i = 0;     // ... similar
    for (int j=j1; j<=j2; j++) {
      t = t0 + dt*j;
      phi = spline(this_phi_c,this_phi_knots,this_phi_n,PHASE_SPLINE_ORDER,&phi_i,t,&local_err);
      if (local_err) {*err=local_err; return;}
#if DEBUG
      if (isnan(this_a_knots[a_i])) {*err=HONK_ERR_NAN;return;}
#endif
      a     = spline(this_a_c,    this_a_knots,    this_a_n,    A_SPLINE_ORDER,&a_i,    t,&local_err);
      if (local_err) {*err=local_err; return;}
      y[j] += a*sin(phi);  // fixme -- add in local memory, copy at end
    }
    this_phi_knots += this_phi_n;
    this_a_knots     += this_a_n;
    this_phi_c += (this_phi_n-1)*(PHASE_SPLINE_ORDER+1); // n-1 because there are no coeffs associated with rightmost knot
    this_a_c     += (this_a_n-1)    *(A_SPLINE_ORDER+1);
  }
  *err = 0;
}

/*
  Evaluate a spline polynomial expressed as an array flattened from the format used by python's PPoly.
  c[j] = flattened version of array c[m][i], with j=(n-1)m+i
  knots[i] = x_i of the ith knot
  n = number of knots
  k = order of polynomial
  i = pointer to initial guess as to the i such that x_i <= x <=x_(i+1); not allowed to be too high, only too low; gets updated
  x = point at which to evaluate the spline
  This is designed to be called repeatedly on the same spline with values of x that are non-decreasing.
  On the first call, *i can be 0, and *i will then be updated automatically.
  Moving to the left past a knot results in an error unless the caller sets *i back to a lower value or 0.
*/
FLOAT spline(__local FLOAT *c,__local FLOAT *knots,int n,int k,int *i,FLOAT x,__local int *err) {
#if DEBUG
  if (*i<0 || *i>n-3) {*err=HONK_INDEX_OUT_OF_RANGE; return NAN;}
#endif
  while (*i<=n-3 && x>knots[*i+1]) {(*i)++;}
  FLOAT d = x-knots[*i];
#if DEBUG
  if (isnan(knots[*i])) {*err=HONK_ERR_NAN; return NAN;}
#endif
  if (d<0) {*err= -1; return 0.0;}
  FLOAT p = 1.0; // (x-x_i)^k-m
  int j = (n-1)*k+*i;
  FLOAT s = 0.0;
  for (int m=k; ; m--) {
    s = s + c[j]*p;
    if (m==0) {break;}
    p = p*d;
    j -= (n-1);
  }
  *err = 0;
  return s;
}

// benchmark using the Riemann zeta function
void fn_zeta(__global FLOAT *y,int i) {
  FLOAT z;
  for (int j=0; j<10; j++) { // repeat the calculation a bunch of times just so it takes enough time to benchmark; 1000 gives about 10 seconds
    z = zeta((FLOAT) (i*0.001+2));
  }
  y[i] = z;
}

/*
  Riemann zeta function, zeta(s) = sum_1^infty 1/n^s.
  For benchmark.
*/
FLOAT zeta(FLOAT s) {
  if (s<=1) {return 0.0;}
  FLOAT zeta = 0.0;
  for (int n=1; n<10000; n++) {
    zeta = zeta + exp(-s*log((FLOAT) n));
  }
  return zeta;
}
