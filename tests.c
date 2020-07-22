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

void tests() {
  FLOAT c[2] = {1.0, 0.0};
  FLOAT knots[2] = {0.0, 1.0};
  int i = 0;
  int err;
  do_one_test(0.3,(double) spline(c,knots,2,1,&i,0.3,&err),"spline, linear",&err);
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
