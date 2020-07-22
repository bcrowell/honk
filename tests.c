/*
  Unit tests of the functions that will run on the GPU. This test can run on either
  the GPU or the CPU, if RUN_ON_CPU is set appropriately in the makefile.
*/

#ifdef RUN_ON_CPU
#include <stdio.h>
#include <math.h>
#endif

#include "gpu.h"
#include "honk.h"

void tests() {
  FLOAT c[2] = {1.0, 0.0};
  FLOAT knots[2] = {0.0, 1.0};
  printf("%lf\n",(double) spline(c,knots,2,1,0,0.3));
}

#ifdef RUN_ON_CPU
int main() {
  printf("Running tests...\n");
  //tests();
  return 0;
}
#endif
