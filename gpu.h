/*
  Allow code to run on the normal CPU rather than the GPU, for testing.
*/
#ifdef RUN_ON_CPU
#define __kernel
#define __global
#define __constant
#define __local
#define __private
#define FLOAT double
#else
#define FLOAT float
#endif
