#ifdef RUN_ON_CPU
#include <math.h>
#endif

#define DO_DEBUGGING 1
#if DO_DEBUGGING
#define DEBUG(x) x
#else
#define DEBUG(x)
#endif

#include "gpu.h"
#include "honk.h"
#include "constants.h"

#ifndef RUN_ON_CPU
__kernel void oscillator(__global FLOAT *y,
                         __global int *err, __global int *error_details, __global FLOAT *info,__global int *n_info,
                         __global const FLOAT *v1, __global const FLOAT *v2, __global const FLOAT *v3, __global const FLOAT *v4,
                         __global const int *k1,  __global const int *k2,
                         __global const long *i_pars, __global const FLOAT *f_pars
                          ) {
  int i = get_global_id(0); // index of the current element in the computational grid
  err[i] = 0; // see oscillator.py for info about how errors are handled
  fn_osc(y,i,err,error_details,info,n_info,v1,v2,v3,v4,k1,k2,i_pars,f_pars);
}

#endif

#define ERR(error_array,instance,err) flag_err(error_array,instance,err,__LINE__)
// ... see oscillator.py for info about how errors are handled

void fn_osc(__global FLOAT *y,int i,
                         __global int *err, __global int *error_details, __global FLOAT *info,__global int *n_info,
                         __global const FLOAT *v1, __global const FLOAT *v2,
                         __global const FLOAT *v3, __global const FLOAT *v4,
                         __global const int *k1,  __global const int *k2,
                         __global const long *i_pars, __global const FLOAT *f_pars) {
  int j1;
  int j2;
  __local int samples_per_instance;
  __local int n_partials;
  __local int n_samples;
  samples_per_instance = i_pars[0];
  n_partials = i_pars[1];
  n_samples = i_pars[2];
  // copy data to local memory for efficiency:
  __local int phi_n[MAX_PARTIALS]; // phi_n[m] is the number of knots in the piecewise polynomial for the mth partial's phase
  __local int a_n[MAX_PARTIALS];
  __local FLOAT phi_knots[MAX_SPLINE_KNOTS];
  __local FLOAT a_knots[MAX_SPLINE_KNOTS];
  __local FLOAT phi_c_local[MAX_SPLINE_COEFFS];
  __local FLOAT a_c_local[MAX_SPLINE_COEFFS];
  DEBUG(if (!(samples_per_instance>0 && n_partials>0)) {ERR(err,i,HONK_ERR_ILLEGAL_VALUE); return;}) // sanity check
  // ---- copy number of knots
  if (n_partials>MAX_PARTIALS) {ERR(err,i,HONK_ERR_TOO_MANY_PARTIALS); return;}
  for (int m=0; m<n_partials; m++) {
    phi_n[m] = k1[m];
    a_n[m]   = k2[m];
    if (phi_n[m]>MAX_SPLINE_KNOTS || a_n[m]>MAX_SPLINE_KNOTS) {ERR(err,i,HONK_ERR_TOO_MANY_KNOTS_IN_SPLINE); return;}
  }
  // ---- copy locations of knots
  int k_phi = 0;
  int k_a = 0;
  for (int m=0; m<n_partials; m++) {
    int this_phi_n = phi_n[m];
    int this_a_n = a_n[m];
    if (k_phi+this_phi_n>MAX_SPLINE_KNOTS || k_a+this_a_n>MAX_SPLINE_KNOTS) {ERR(err,i,HONK_ERR_TOO_MANY_KNOTS_IN_SPLINE); return;}
    for (int j=0; j<this_phi_n; j++) {
      DEBUG(if (k_phi+j>MAX_SPLINE_KNOTS) {ERR(err,i,HONK_ERR_INDEX_OUT_OF_RANGE);return;})
      phi_knots[k_phi+j] = v2[k_phi+j];
    }
    for (int j=0; j<this_a_n; j++) {
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
    if (k_phi+phi_size>MAX_SPLINE_COEFFS || k_a+a_size>MAX_SPLINE_COEFFS) {ERR(err,i,HONK_ERR_SPLINE_TOO_LARGE); return;}
    for (int j=0; j<phi_size; j++) {
      phi_c_local[k_phi+j] = v1[k_phi+j];
    }
    for (int j=0; j<a_size; j++) {
      a_c_local[k_a+j]     = v3[k_a+j];
    }
    k_phi += phi_size;
    k_a     += a_size;
  }
  j1 = i*samples_per_instance;
  j2 = (i+1)*samples_per_instance-1;
  DEBUG(if (!(samples_per_instance>0)) {ERR(err,i,HONK_ERR_ILLEGAL_VALUE); return;})
  DEBUG(if (!(j1>=0)) {ERR(err,i,HONK_ERR_ILLEGAL_VALUE); return;})
  DEBUG(if (!(j2>=0)) {ERR(err,i,HONK_ERR_ILLEGAL_VALUE); return;})
  DEBUG(if (!(j2>=j1)) {set_flags(error_details,i,j1,j2,samples_per_instance); ERR(err,i,HONK_ERR_ILLEGAL_VALUE); return;})
  DEBUG(if (!(j1>=0 && j2>=0 && j2>=j1 && samples_per_instance>0)) {set_flags(error_details,i,j1,(int) sizeof(j1),samples_per_instance); ERR(err,i,HONK_ERR_ILLEGAL_VALUE); return;}) // sanity check
  DEBUG(if (!(j1>=0 && j2>=0 && j2>=j1)) {ERR(err,i,HONK_ERR_ILLEGAL_VALUE); return;}) // sanity check
  DEBUG(if (j2>=n_samples) {ERR(err,i,HONK_ERR_INDEX_OUT_OF_RANGE); return;}) // sanity check
  oscillator_cubic_spline(y,err,error_details,i,
                          phi_c_local,phi_knots,phi_n,
                          a_c_local,    a_knots    ,a_n,
                          f_pars[0],f_pars[1],j1,j2,n_partials
                         );
}


void oscillator_cubic_spline(__global FLOAT *y,__global int *err,__global int *error_details,int instance,
                             __local FLOAT *phi_c,__local FLOAT *phi_knots,__local int *phi_n,
                             __local FLOAT *a_c,    __local FLOAT *a_knots,    __local int *a_n,
                             FLOAT t0,FLOAT dt,int j1,int j2,int n_partials) {
  FLOAT phi,a,t;
  for (int j=j1; j<=j2; j++) {
    y[j] = 0.0; // y is write-only, so it can't be initialized to zero for us
        // ... fixme -- inefficient
  }
  __local FLOAT *this_phi_c = phi_c;
  __local FLOAT *this_a_c     = a_c;
  __local FLOAT *this_phi_knots = phi_knots;
  __local FLOAT *this_a_knots = a_knots;
  for (int m=0; m<n_partials; m++) {
    int this_phi_n = phi_n[m];
    int this_a_n = a_n[m];
    int phi_i = 0; // current knot number in spline for phi
    int a_i = 0;     // ... similar
    for (int j=j1; j<=j2; j++) {
      t = t0 + dt*j;
      int local_err; 
      phi = spline(this_phi_c,this_phi_knots,this_phi_n,PHASE_SPLINE_ORDER,&phi_i,t,&local_err);
      if (local_err) {ERR(err,instance,local_err); return;}
      DEBUG(if (isnan(this_a_knots[a_i])) {ERR(err,instance,HONK_ERR_NAN); return;})
      a     = spline(this_a_c,    this_a_knots,    this_a_n,    A_SPLINE_ORDER,&a_i,    t,&local_err);
      if (local_err) {ERR(err,instance,local_err); return;}
      y[j] += a*sin(phi);  // fixme -- add in local memory, copy at end
    }
    this_phi_knots += this_phi_n;
    this_a_knots     += this_a_n;
    this_phi_c += (this_phi_n-1)*(PHASE_SPLINE_ORDER+1); // n-1 because there are no coeffs associated with rightmost knot
    this_a_c     += (this_a_n-1)    *(A_SPLINE_ORDER+1);
  }
}

/*
  Evaluate a spline polynomial expressed as an array flattened from the format used by python's PPoly.
  c[j] = flattened version of array c[m][i], with j=(n-1)m+i 
    This mimicks the weird way the python PPoly object stores the coefficients.
    There are only n-1 polynomials, because polynomial i covers the interval to the right of x_i, and nothing is to the right of the last knot.
  knots[i] = x_i of the ith knot
  n = number of knots
  k = order of polynomial
  m = k-exponent
  i = pointer to initial guess as to the i such that x_i <= x <=x_(i+1); not allowed to be too high, only too low; gets updated
  x = point at which to evaluate the spline
  This is designed to be called repeatedly on the same spline with values of x that are non-decreasing.
  On the first call, *i can be 0, and *i will then be updated automatically.
  Moving to the left past a knot results in an error unless the caller sets *i back to a lower value or 0.
*/
FLOAT spline(__local FLOAT *c,__local FLOAT *knots,int n,int k,int *i,FLOAT x,int *local_err) {
  // In the following code, the idea is that i=n-1 is not legal.
  DEBUG(if (*i<0 || *i>=n-1) {*local_err = HONK_ERR_INDEX_OUT_OF_RANGE; return NAN;})
  while (*i<=n-3 && x>knots[*i+1]) {(*i)++;} // i<n-3 means that i+1<n-1, which would be legal
  FLOAT d = x-knots[*i];
  DEBUG(if (isnan(knots[*i])) {*local_err = HONK_ERR_NAN; return NAN;})
  if (d<0) {*local_err= HONK_ERR_ILLEGAL_VALUE; return 0.0;}
  FLOAT p = 1.0; // (x-x_i)^k-m
  int j = (n-1)*k+*i;
  FLOAT s = 0.0;
  for (int m=k; ; m--) {
    s = s + c[j]*p;
    if (m==0) {break;}
    p = p*d;
    j -= (n-1);
  }
  *local_err = 0;
  return s;
}

void set_flags(__global int *error_array,int instance,int f1,int f2,int f3) {
  int k=instance*64;
  error_array[k] = f1;
  error_array[k+1] = f2;
  error_array[k+2] = f3;
}

// See oscillator.py for info about how errors are handled.
void flag_err(__global int *error_array,int instance,int err,int where_in_code) {
  error_array[instance] = err*1000+where_in_code;
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
