#define FLOAT float

FLOAT zeta(FLOAT s);
FLOAT spline(FLOAT *c,FLOAT *knots,int n,int k,int i,FLOAT x);

__kernel void oscillator(__global FLOAT *y, __global const FLOAT *a) {
  // Get the index of the current element to be processed
  int i = get_global_id(0);
  FLOAT z;
  for (int j=0; j<10; j++) { // repeat the calculation a bunch of times just so it takes enough time to benchmark; 1000 gives about 10 seconds
    z = zeta((FLOAT) (i*0.001+2));
  }
  y[i] = z;
}

/*
  Evaluate a spline polynomial expressed as an array flattened from the format used by python's PPoly.
  c[j] = flattened version of array c[m][i], with j=(n-1)m+i
  knots[i] = x_i of the ith knot
  n = number of knots
  k = order of polynomial
  i = initial guess as to the i such that x_i <= x <=x_(i+1)
  x = point at which to evaluate the spline
*/
FLOAT spline(FLOAT *c,FLOAT *knots,int n,int k,int i,FLOAT x) {
}

/*
  Riemann zeta function, zeta(s) = sum_1^infty 1/n^s.
  For benchmark.
*/
FLOAT zeta(FLOAT s) {
  if (s<=1) {return 0.0;}
  FLOAT zeta = 0.0;
  for (int n=1; n<10000; n++) {
    zeta = zeta + exp(-s*log((FLOAT) n));
  }
  return zeta;
}
