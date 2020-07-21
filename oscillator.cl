__kernel void oscillator(__global const int *a, __global const int *b, __global int *c) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    c[i] = 24;
}
