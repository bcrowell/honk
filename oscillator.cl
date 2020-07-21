__kernel void oscillator(__global const int *a, __global const float *b, __global int *c) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);

    int n = a[0];

    float x = b[0];

    c[i] = (n+i)*x;
 
}
