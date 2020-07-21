float zeta(float s);

__kernel void oscillator(__global float *y, __global const float *a) {
  // Get the index of the current element to be processed
  int i = get_global_id(0);
  float z;
  for (int j=0; j<100; j++) { // repeat the calculation a bunch of times just so it takes enough time to benchmark
    z = zeta((float) (i*0.001+2));
  }
  y[i] = z;
}



/*
  Riemann zeta function, zeta(s) = sum_1^infty 1/n^s.
  For benchmark.
*/
float zeta(float s) {
  if (s<=1) {return 0.0;}
  float zeta = 0.0;
  for (int n=1; n<10000; n++) {
    zeta = zeta + exp(-s*log((float) n));
  }
  return zeta;
}
