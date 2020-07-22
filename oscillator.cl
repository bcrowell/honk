#ifdef RUN_ON_CPU
#include <math.h>
#endif

#include "gpu.h"
#include "honk.h"

#ifndef RUN_ON_CPU
__kernel void oscillator(__global const int function,
                         __global FLOAT *y,
                         __global int *err, __global FLOAT *info,__global int *n_info,
                         __global const FLOAT *v1, __global const FLOAT *v2, __global const FLOAT *v3, __global const FLOAT *v4,
                         __global const long *i_pars, __global const FLOAT *f_pars
                          ) {
  int i = get_global_id(0); // index of the current element in the computational grid
  if (function==HONK_FN_ZETA) {fn_zeta(y,i); *err=0; return;}
  *err = HONK_ERR_UNDEFINED_FN;
}
#endif

// benchmark using the Riemann zeta function
void fn_zeta(FLOAT *y,int i) {
  FLOAT z;
  for (int j=0; j<10; j++) { // repeat the calculation a bunch of times just so it takes enough time to benchmark; 1000 gives about 10 seconds
    z = zeta((FLOAT) (i*0.001+2));
  }
  y[i] = z;
}

void oscillator_cubic_spline(FLOAT *y,
                             FLOAT *omega_c,FLOAT *omega_knots,int omega_n,
                             FLOAT *a_c,FLOAT *a_knots,int a_n,
                             FLOAT phase,
                             FLOAT t0,FLOAT dt,int j1,int j2,int *err) {
  int omega_i = 0;
  int a_i = 0;
  FLOAT omega,a,t;
  for (int j=j1; j<=j2; j++) {
    t = t0 + dt*j;
    omega = spline(omega_c,omega_knots,omega_n,3,&omega_i,t,err);
    if (err) {return;}
    a     = spline(a_c,    a_knots,    a_n,    3,&a_i,    t,err);
    if (err) {return;}
    y[j] = a*sin(omega*t+phase);
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
FLOAT spline(FLOAT *c,FLOAT *knots,int n,int k,int *i,FLOAT x,int *err) {
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
