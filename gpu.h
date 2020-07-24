/*
  Allow code to run either on the GPU or on the normal CPU, for testing.
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
