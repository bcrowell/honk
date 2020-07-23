/*
  Unit tests of the functions that will run on the GPU. This test can run on either
  the GPU or the CPU, if RUN_ON_CPU is set appropriately in the makefile.
*/

#ifdef RUN_ON_CPU
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#endif

#include "gpu.h"
#include "honk.h"

void do_one_test(FLOAT expected,FLOAT actual,char *description,int *err);
void test_spline_1();
void test_spline_2();
void benchmark();

void tests() {
  test_spline_1();
  test_spline_2();
  // benchmark();
}

#define N_BENCHMARK 1024000
void benchmark() {
  FLOAT y[N_BENCHMARK];
  int i=0;
  int err;
  FLOAT info[10];
  int n_info[10];
  FLOAT v1[10],v2[10],v3[10],v4[10];
  long i_pars[10];
  FLOAT f_pars[10];
  FLOAT sample_freq = 44100.0;
  v2[0] = 0.0; v2[1] = 1.0;
  v4[0] = 0.0; v4[1] = 1.0;
  // constant spline polynomials
  v1[3] = 1000.0*2*3.141; v1[4] = v1[3]; // constant omega
  v3[3] = 1.0; v3[4] = 1.0; // constant amplitude=1
  // scalar parameters:
  i_pars[0] = 2; // n for omega spline
  i_pars[1] = 2; // n for amplitude spline
  i_pars[2] = N_BENCHMARK; // samples per instance, on CPU
  f_pars[0] = 0.0; // phase
  f_pars[1] = 0.0; // t0
  f_pars[2] = 1/sample_freq; // dt
  for (int j=1; j<=100; j++) { // repeat for benchmark
    fn_osc(y,i,&err,info,n_info,v1,v2,v3,v4,i_pars,f_pars);
  }
}

void test_spline_1() {
  FLOAT c[2] = {1.0, 0.0};
  FLOAT knots[2] = {0.0, 1.0};
  int i = 0;
  int err;
  do_one_test((FLOAT) 0.3,(FLOAT) spline(c,knots,2,1,&i,0.3,&err),"spline, linear",&err);
}

void test_spline_2() {
  /*
    python code to generate this test's data:
    from scipy import interpolate 
    x_points = [ 0, 1, 2, 3, 4]
    y_points = [ 0, 1, 16, 81, 256]
    p = interpolate.CubicSpline(x_points, y_points)
    print(p(2.7))
    print(p.c.flatten())
  */
  FLOAT c[16] = {5.0,  5.0, 11.0, 11.0, -8.0,  7.0, 22.0, 55.0,  4.0,  3.0, 32.0, 109.0,  0.0,  1.0, 16.0, 81.0};
  FLOAT knots[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
  int i = 0;
  int err;
  do_one_test((FLOAT) 52.953,(FLOAT) spline(c,knots,5,3,&i,2.7,&err),"spline, cubic, approximating y=x^4",&err);
}

void do_one_test(FLOAT expected,FLOAT actual,char *description,int *err) {
  if (fabs(expected-actual)<1.0e-6 && *err==0) {
    printf("passed: %s\n",description);
  }
  else {
    printf("failed: %s, expected=%lf, actual=%lf, err=%d\n",description,(double) expected,(double) actual,(int) *err);
    exit(-1);
  }
}

#ifdef RUN_ON_CPU
int main(void) {
  printf("Running tests...\n");
  tests();
  return 0;
}
#endif
