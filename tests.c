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

void do_one_test(FLOAT expected,FLOAT actual,char *description);

void tests() {
  FLOAT c[2] = {1.0, 0.0};
  FLOAT knots[2] = {0.0, 1.0};
  int i = 0;
  do_one_test(0.3,(double) spline(c,knots,2,1,&i,0.3),"spline, linear");
}

void do_one_test(FLOAT expected,FLOAT actual,char *description) {
  if (fabs(expected-actual)<1.0e-6) {
    printf("passed: %s\n",description);
  }
  else {
    printf("failed: %s\n",description);
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
