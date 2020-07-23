#ifdef RUN_ON_CPU
#include <math.h>
#endif

#include "gpu.h"
#include "honk.h"

#ifndef RUN_ON_CPU
__kernel void oscillator(__global const int *fn,
                         __global FLOAT *y,
                         __global int *err, __global FLOAT *info,__global int *n_info,
                         __global const FLOAT *v1, __global const FLOAT *v2, __global const FLOAT *v3, __global const FLOAT *v4,
                         __global const int *k1,  __global const int *k2,
                         __global const long *i_pars, __global const FLOAT *f_pars
                          ) {
  int i = get_global_id(0); // index of the current element in the computational grid
  *err = 0;
  if (*fn==HONK_FN_ZETA) {fn_zeta(y,i); return;}
  if (*fn==HONK_FN_OSC) {fn_osc(y,i,err,info,n_info,v1,v2,v3,v4,k1,k2,i_pars,f_pars); return;}
  *err = HONK_ERR_UNDEFINED_FN;
}
#endif

#define MAX_SPLINE_KNOTS 80
#define SPLINE_ORDER 3
#define MAX_SPLINE_COEFFS (MAX_SPLINE_KNOTS*(SPLINE_ORDER+1))
// ... total number of cubic spline coefficients in all partials
#define MAX_PARTIALS 16

void fn_osc(__global FLOAT *y,int i,
                         __global int *err, __global FLOAT *info,__global int *n_info,
                         __global const FLOAT *v1, __global const FLOAT *v2,
                         __global const FLOAT *v3, __global const FLOAT *v4,
                         __global const int *k1,  __global const int *k2,
                         __global const long *i_pars, __global const FLOAT *f_pars) {
  int samples_per_instance = i_pars[0];
  int n_partials = i_pars[1];
  // copy data to local memory for efficiency:
  __local int omega_n[MAX_PARTIALS];
  __local int a_n[MAX_PARTIALS];
  __local FLOAT omega_knots[MAX_SPLINE_KNOTS];
  __local FLOAT a_knots[MAX_SPLINE_KNOTS];
  __local FLOAT omega_c_local[MAX_SPLINE_COEFFS];
  __local FLOAT a_c_local[MAX_SPLINE_COEFFS];
  // ---- copy number of knots
  if (n_partials>MAX_PARTIALS) {*err=HONK_ERR_TOO_MANY_PARTIALS; return;}
  for (int m=0; m<n_partials; m++) {
    omega_n[m] = k1[m];
    a_n[m]     = k2[m];
  }
  // ---- copy locations of knots
  int k_omega = 0;
  int k_a = 0;
  for (int m=0; m<n_partials; m++) {
    int this_omega_n = omega_n[m];
    int this_a_n = a_n[m];
    if (k_omega+this_omega_n>MAX_SPLINE_KNOTS || k_a+this_a_n>MAX_SPLINE_KNOTS) {*err=HONK_ERR_TOO_MANY_KNOTS_IN_SPLINE; return;}
    for (int j=0; j<k_omega+this_omega_n; j++) {
      omega_knots[k_omega+j] = v2[k_omega+j];
    }
    for (int j=0; j<k_omega+this_omega_n; j++) {
      a_knots[k_a+j]         = v4[k_a+j];
    }
    k_omega += this_omega_n;
    k_a     += this_a_n;
  }
  // ---- copy coefficients of cubic spline polynomials
  k_omega = 0;
  k_a = 0;
  for (int m=0; m<n_partials; m++) {
    int this_omega_n = omega_n[m];
    int this_a_n = a_n[m];
    int omega_size = (this_omega_n-1)*(SPLINE_ORDER+1); // n-1 because there are no coeffs associated with rightmost knot
    int a_size =     (this_a_n-1)    *(SPLINE_ORDER+1);
    if (k_omega+omega_size>MAX_SPLINE_COEFFS || k_a+a_size>MAX_SPLINE_COEFFS) {*err=HONK_ERR_SPLINE_TOO_LARGE; return;}
    for (int j=0; j<k_omega+omega_size; j++) {
      omega_c_local[k_omega+j] = v1[k_omega+j];
    }
    for (int j=0; j<k_a+a_size; j++) {
      omega_c_local[k_a+j] = v3[k_a+j];
    }
    k_omega += omega_size;
    k_a     += a_size;
  }
  oscillator_cubic_spline(y,
                          omega_c_local,omega_knots,omega_n,
                          a_c_local    ,a_knots    ,a_n,
                          f_pars[0],f_pars[1],f_pars[2],i*samples_per_instance,(i+1)*samples_per_instance-1,n_partials,
                          err
                         );
}


void oscillator_cubic_spline(__global FLOAT *y,
                             __local FLOAT *omega_c,__local FLOAT *omega_knots,__local int *omega_n,
                             __local FLOAT *a_c,    __local FLOAT *a_knots,    __local int *a_n,
                             FLOAT phase,FLOAT t0,FLOAT dt,int j1,int j2,int n_partials,
                             __global int *err) {
  FLOAT omega,a,t;
  for (int j=j1; j<=j2; j++) {
    y[j] = 0.0; // fixme -- inefficient
  }
  __local FLOAT *this_omega_c = omega_c;
  __local FLOAT *this_a_c     = a_c;
  __local FLOAT *this_omega_knots = omega_knots;
  __local FLOAT *this_a_knots = a_knots;
  __local int local_err; // for efficiency, avoid accessing the global err
  for (int m=0; m<n_partials; m++) {
    int this_omega_n = omega_n[m];
    int this_a_n = a_n[m];
    int omega_i = 0; // current knot number in spline for omega
    int a_i = 0;     // ... similar
    for (int j=j1; j<=j2; j++) {
      t = t0 + dt*j;
      omega = spline(this_omega_c,this_omega_knots,this_omega_n,SPLINE_ORDER,&omega_i,t,&local_err);
      if (local_err) {*err=local_err; return;}
      a     = spline(this_a_c,    this_a_knots,    this_a_n,    SPLINE_ORDER,&a_i,    t,&local_err);
      if (local_err) {*err=local_err; return;}
      // qwe -------
         omega=1000.0; a=1.0;
      y[j] += a*sin(omega*t+phase);  // fixme -- add in local memory, copy at end
    }
    this_omega_knots += this_omega_n;
    this_a_knots     += this_a_n;
    this_omega_c += (this_omega_n-1)*(SPLINE_ORDER+1); // n-1 because there are no coeffs associated with rightmost knot
    this_a_c     += (this_a_n-1)    *(SPLINE_ORDER+1);
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
  while (*i<=n-3 && x>knots[*i+1]) {(*i)++;}
  FLOAT d = x-knots[*i];
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
