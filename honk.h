FLOAT spline(__local FLOAT *c,__local FLOAT *knots,int n,int k,int *i,FLOAT x,__local int *err);
void fn_osc(__global FLOAT *y,int i,
                         __global int *err, __global FLOAT *info,__global int *n_info,
                         __global const FLOAT *v1, __global const FLOAT *v2, __global const FLOAT *v3, __global const FLOAT *v4,
                         __global const int *k1,  __global const int *k2,
                         __global const long *i_pars, __global const FLOAT *f_pars);
void oscillator_cubic_spline(__global FLOAT *y,
                             __local FLOAT *omega_c,__local FLOAT *omega_knots,__local int *omega_n,
                             __local FLOAT *a_c,    __local FLOAT *a_knots,    __local int *a_n,
                             FLOAT t0,FLOAT dt,int j1,int j2,int n_partials,
                             __global int *err);
void fn_zeta(__global FLOAT *y,int i);
FLOAT zeta(FLOAT s);

