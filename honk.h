FLOAT spline(__global const FLOAT *c,__global const FLOAT *knots,int n,int k,int *i,FLOAT x,__global int *err);
void fn_osc(__global FLOAT *y,int i,
                         __global int *err, __global FLOAT *info,__global int *n_info,
                         __global const FLOAT *v1, __global const FLOAT *v2, __global const FLOAT *v3, __global const FLOAT *v4,
                         __global const int *k1,  __global const int *k2,
                         __global const long *i_pars, __global const FLOAT *f_pars);
void oscillator_cubic_spline(__global FLOAT *y,
                             __global const FLOAT *omega_c,__global const FLOAT *omega_knots,__global const int *omega_n,
                             __global const FLOAT *a_c,__global const FLOAT *a_knots,__global const int *a_n,
                             FLOAT phase,FLOAT t0,FLOAT dt,int j1,int j2,int n_partials,
                             __global int *err);
void fn_zeta(__global FLOAT *y,int i);
FLOAT zeta(FLOAT s);

#define HONK_FN_OSC 1
#define HONK_FN_ZETA 2

#define HONK_ERR_UNDEFINED_FN 1