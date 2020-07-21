float zeta(float s);

__kernel void oscillator(__global float *y, __global const float *a) {
  // Get the index of the current element to be processed
  int i = get_global_id(0);
  y[i] = zeta((float) i);
}

// for Riemann zeta function, zeta(s) = sum_1^infty 1/n^s
float zeta(float s) {
  if (s<=1) {return 0.0;}
  float zeta = 0.0;
  for (int n=1; n<1000; n++) {
    zeta = zeta + exp(-s*log((float) n));
  }
  return zeta;
}
