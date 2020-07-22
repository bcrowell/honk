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

void tests() {
  test_spline_1();
  test_spline_2();
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
