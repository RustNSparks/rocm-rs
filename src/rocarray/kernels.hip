// src/rocarray/kernels.hip
#include <hip/hip_runtime.h>

// Macro to generate elementwise operations for different types
#define DEFINE_ELEMENTWISE_OP(op_name, op_symbol, type, type_suffix) \
extern "C" __global__ void elementwise_##op_name##_##type_suffix( \
    const type* a, const type* b, type* result, unsigned int n) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < n) { \
        result[idx] = a[idx] op_symbol b[idx]; \
    } \
}

#define DEFINE_SCALAR_OP(op_name, op_symbol, type, type_suffix) \
extern "C" __global__ void scalar_##op_name##_##type_suffix( \
    const type* input, type scalar, type* result, unsigned int n) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < n) { \
        result[idx] = input[idx] op_symbol scalar; \
    } \
}

#define DEFINE_REDUCE_SUM(type, type_suffix) \
extern "C" __global__ void reduce_sum_##type_suffix( \
    const type* input, unsigned int n, type* result) { \
    __shared__ type sdata[256]; \
    int tid = threadIdx.x; \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    \
    sdata[tid] = (idx < n) ? input[idx] : (type)0; \
    __syncthreads(); \
    \
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { \
        if (tid < s) { \
            sdata[tid] += sdata[tid + s]; \
        } \
        __syncthreads(); \
    } \
    \
    if (tid == 0) { \
        atomicAdd(result, sdata[0]); \
    } \
}

#define DEFINE_REDUCE_MAX(type, type_suffix) \
extern "C" __global__ void reduce_max_##type_suffix( \
    const type* input, unsigned int n, type* result) { \
    __shared__ type sdata[256]; \
    int tid = threadIdx.x; \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    \
    sdata[tid] = (idx < n) ? input[idx] : input[0]; \
    __syncthreads(); \
    \
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { \
        if (tid < s) { \
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]); \
        } \
        __syncthreads(); \
    } \
    \
    if (tid == 0) { \
        atomicMax((int*)result, *(int*)&sdata[0]); \
    } \
}

#define DEFINE_REDUCE_MIN(type, type_suffix) \
extern "C" __global__ void reduce_min_##type_suffix( \
    const type* input, unsigned int n, type* result) { \
    __shared__ type sdata[256]; \
    int tid = threadIdx.x; \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    \
    sdata[tid] = (idx < n) ? input[idx] : input[0]; \
    __syncthreads(); \
    \
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { \
        if (tid < s) { \
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]); \
        } \
        __syncthreads(); \
    } \
    \
    if (tid == 0) { \
        atomicMin((int*)result, *(int*)&sdata[0]); \
    } \
}

#define DEFINE_RANGE_FILL(type, type_suffix) \
extern "C" __global__ void generic_range_##type_suffix( \
    type start, type step, unsigned int n, type* output) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < n) { \
        output[idx] = start + (type)idx * step; \
    } \
}

// Generate kernels for all supported types
DEFINE_ELEMENTWISE_OP(add, +, float, float)
DEFINE_ELEMENTWISE_OP(sub, -, float, float)
DEFINE_ELEMENTWISE_OP(mul, *, float, float)
DEFINE_ELEMENTWISE_OP(div, /, float, float)

DEFINE_ELEMENTWISE_OP(add, +, double, double)
DEFINE_ELEMENTWISE_OP(sub, -, double, double)
DEFINE_ELEMENTWISE_OP(mul, *, double, double)
DEFINE_ELEMENTWISE_OP(div, /, double, double)

DEFINE_ELEMENTWISE_OP(add, +, int, int)
DEFINE_ELEMENTWISE_OP(sub, -, int, int)
DEFINE_ELEMENTWISE_OP(mul, *, int, int)
DEFINE_ELEMENTWISE_OP(div, /, int, int)

DEFINE_ELEMENTWISE_OP(add, +, unsigned int, uint)
DEFINE_ELEMENTWISE_OP(sub, -, unsigned int, uint)
DEFINE_ELEMENTWISE_OP(mul, *, unsigned int, uint)
DEFINE_ELEMENTWISE_OP(div, /, unsigned int, uint)

DEFINE_ELEMENTWISE_OP(add, +, long long, long)
DEFINE_ELEMENTWISE_OP(sub, -, long long, long)
DEFINE_ELEMENTWISE_OP(mul, *, long long, long)
DEFINE_ELEMENTWISE_OP(div, /, long long, long)

DEFINE_ELEMENTWISE_OP(add, +, unsigned long long, ulong)
DEFINE_ELEMENTWISE_OP(sub, -, unsigned long long, ulong)
DEFINE_ELEMENTWISE_OP(mul, *, unsigned long long, ulong)
DEFINE_ELEMENTWISE_OP(div, /, unsigned long long, ulong)

// Scalar operations
DEFINE_SCALAR_OP(add, +, float, float)
DEFINE_SCALAR_OP(mul, *, float, float)
DEFINE_SCALAR_OP(add, +, double, double)
DEFINE_SCALAR_OP(mul, *, double, double)
DEFINE_SCALAR_OP(add, +, int, int)
DEFINE_SCALAR_OP(mul, *, int, int)
DEFINE_SCALAR_OP(add, +, unsigned int, uint)
DEFINE_SCALAR_OP(mul, *, unsigned int, uint)
DEFINE_SCALAR_OP(add, +, long long, long)
DEFINE_SCALAR_OP(mul, *, long long, long)
DEFINE_SCALAR_OP(add, +, unsigned long long, ulong)
DEFINE_SCALAR_OP(mul, *, unsigned long long, ulong)

// Reduction operations
DEFINE_REDUCE_SUM(float, float)
DEFINE_REDUCE_SUM(double, double)
DEFINE_REDUCE_SUM(int, int)
DEFINE_REDUCE_SUM(unsigned int, uint)
DEFINE_REDUCE_SUM(long long, long)
DEFINE_REDUCE_SUM(unsigned long long, ulong)

DEFINE_REDUCE_MAX(float, float)
DEFINE_REDUCE_MAX(double, double)
DEFINE_REDUCE_MAX(int, int)
DEFINE_REDUCE_MAX(unsigned int, uint)

DEFINE_REDUCE_MIN(float, float)
DEFINE_REDUCE_MIN(double, double)
DEFINE_REDUCE_MIN(int, int)
DEFINE_REDUCE_MIN(unsigned int, uint)

// Range operations
DEFINE_RANGE_FILL(float, float)
DEFINE_RANGE_FILL(double, double)
DEFINE_RANGE_FILL(int, int)
DEFINE_RANGE_FILL(unsigned int, uint)
DEFINE_RANGE_FILL(long long, long)
DEFINE_RANGE_FILL(unsigned long long, ulong)

// Linspace for double precision
extern "C" __global__ void linspace_double(
    double start, double step, unsigned int n, double* output) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        output[idx] = start + (double)idx * step;
    }
}

// Generic map kernel (placeholder - would need runtime compilation for arbitrary functions)
extern "C" __global__ void generic_map(
    const void* input, void* output, unsigned int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        // This is a placeholder - real implementation would need function pointers or runtime compilation
        // For now, just copy the data
        ((float*)output)[idx] = ((const float*)input)[idx];
    }
}

// Generic filter kernel (placeholder)
extern "C" __global__ void generic_filter(
    const void* input, void* output, unsigned int n, unsigned int* count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        // Placeholder filter - just copy all elements for now
        ((float*)output)[idx] = ((const float*)input)[idx];
        if (idx == 0) *count = n;
    }
}

// Generic reduce kernel (placeholder)
extern "C" __global__ void generic_reduce(
    const void* input, unsigned int n, void* initial, void* result) {
    // Placeholder - would need runtime compilation for arbitrary reduction functions
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *((float*)result) = *((const float*)initial);
    }
}

// Generic search kernel (placeholder)
extern "C" __global__ void generic_search(
    const void* input, unsigned int n, int* result) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        // Placeholder search - return first index for now
        if (idx == 0) *result = 0;
    }
}

// Utility kernels
extern "C" __global__ void reverse_array_float(float* data, unsigned int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n / 2) {
        float temp = data[idx];
        data[idx] = data[n - 1 - idx];
        data[n - 1 - idx] = temp;
    }
}

extern "C" __global__ void reverse_array_double(double* data, unsigned int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n / 2) {
        double temp = data[idx];
        data[idx] = data[n - 1 - idx];
        data[n - 1 - idx] = temp;
    }
}

extern "C" __global__ void reverse_array_int(int* data, unsigned int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n / 2) {
        int temp = data[idx];
        data[idx] = data[n - 1 - idx];
        data[n - 1 - idx] = temp;
    }
}

extern "C" __global__ void reverse_array_uint(unsigned int* data, unsigned int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n / 2) {
        unsigned int temp = data[idx];
        data[idx] = data[n - 1 - idx];
        data[n - 1 - idx] = temp;
    }
}