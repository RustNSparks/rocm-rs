
#include <hip/hip_runtime.h>

extern "C" __global__ void vector_add(const float* a, const float* b, float* c, unsigned int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}