FLOAT zeta(FLOAT s);
FLOAT spline(FLOAT *c,FLOAT *knots,int n,int k,int *i,FLOAT x,int *err);
void oscillator_cubic_spline(FLOAT *y,
                             FLOAT *omega_c,FLOAT *omega_knots,int omega_n,
                             FLOAT *a_c,FLOAT *a_knots,int a_n,
                             FLOAT phase,
                             FLOAT t0,FLOAT dt,int j1,int j2,int *err);
