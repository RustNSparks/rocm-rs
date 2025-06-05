// src/rocarray/kernels.hip - Enhanced HIP kernels with broadcasting and advanced operations
#include <hip/hip_runtime.h>

// =============================================================================
// Utility functions and macros
// =============================================================================

// Helper function to compute multidimensional index from flat index
__device__ inline void unravel_index(unsigned int flat_idx, const unsigned int* dims,
                                    unsigned int ndim, unsigned int* indices) {
    unsigned int remaining = flat_idx;
    for (int i = ndim - 1; i >= 0; i--) {
        unsigned int stride = 1;
        for (int j = i + 1; j < ndim; j++) {
            stride *= dims[j];
        }
        indices[i] = remaining / stride;
        remaining %= stride;
    }
}

// Helper function to compute flat index from multidimensional indices
__device__ inline unsigned int ravel_index(const unsigned int* indices, const unsigned int* strides,
                                          unsigned int ndim) {
    unsigned int flat_idx = 0;
    for (int i = 0; i < ndim; i++) {
        flat_idx += indices[i] * strides[i];
    }
    return flat_idx;
}

// Helper function for broadcasting
__device__ inline unsigned int broadcast_index(unsigned int result_idx,
                                              const unsigned int* result_dims, unsigned int result_ndim,
                                              const unsigned int* array_dims, const unsigned int* array_strides,
                                              unsigned int array_ndim) {
    unsigned int result_indices[8]; // Max 8 dimensions
    unsigned int array_indices[8];

    // Unravel result index
    unravel_index(result_idx, result_dims, result_ndim, result_indices);

    // Map to array indices with broadcasting
    for (int i = 0; i < array_ndim; i++) {
        int result_dim_idx = result_ndim - array_ndim + i;
        if (result_dim_idx >= 0) {
            unsigned int dim_size = array_dims[i];
            array_indices[i] = (dim_size == 1) ? 0 : result_indices[result_dim_idx];
        } else {
            array_indices[i] = 0;
        }
    }

    return ravel_index(array_indices, array_strides, array_ndim);
}

// =============================================================================
// Basic element-wise operations
// =============================================================================

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

// =============================================================================
// Broadcasting operations
// =============================================================================

#define DEFINE_ELEMENTWISE_BROADCAST_OP(op_name, op_symbol, type, type_suffix) \
extern "C" __global__ void elementwise_##op_name##_broadcast_##type_suffix( \
    const type* a, const type* b, type* result, \
    const unsigned int* a_dims, const unsigned int* a_strides, unsigned int a_ndim, \
    const unsigned int* b_dims, const unsigned int* b_strides, unsigned int b_ndim, \
    const unsigned int* result_dims, unsigned int result_ndim, unsigned int total_elements) { \
    \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < total_elements) { \
        unsigned int a_idx = broadcast_index(idx, result_dims, result_ndim, a_dims, a_strides, a_ndim); \
        unsigned int b_idx = broadcast_index(idx, result_dims, result_ndim, b_dims, b_strides, b_ndim); \
        result[idx] = a[a_idx] op_symbol b[b_idx]; \
    } \
}

// =============================================================================
// Reduction operations
// =============================================================================

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

#define DEFINE_REDUCE_MAX(type, type_suffix, atomic_op) \
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
        atomic_op(result, sdata[0]); \
    } \
}

#define DEFINE_REDUCE_MIN(type, type_suffix, atomic_op) \
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
        atomic_op(result, sdata[0]); \
    } \
}

// Axis-specific reduction
#define DEFINE_REDUCE_SUM_AXIS(type, type_suffix) \
extern "C" __global__ void reduce_sum_axis_##type_suffix( \
    const type* input, type* output, \
    const unsigned int* dims, const unsigned int* strides, unsigned int ndim, \
    unsigned int axis, unsigned int axis_size, unsigned int output_size) { \
    \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < output_size) { \
        type sum = (type)0; \
        \
        /* Compute base index without the reduced axis */ \
        unsigned int base_idx = 0; \
        unsigned int temp_idx = idx; \
        \
        for (int dim = ndim - 1; dim >= 0; dim--) { \
            if (dim != axis) { \
                unsigned int dim_stride = 1; \
                for (int j = dim + 1; j < ndim; j++) { \
                    if (j != axis) dim_stride *= dims[j]; \
                } \
                unsigned int coord = temp_idx / dim_stride; \
                temp_idx %= dim_stride; \
                base_idx += coord * strides[dim]; \
            } \
        } \
        \
        /* Sum along the specified axis */ \
        for (int i = 0; i < axis_size; i++) { \
            unsigned int input_idx = base_idx + i * strides[axis]; \
            sum += input[input_idx]; \
        } \
        \
        output[idx] = sum; \
    } \
}

// =============================================================================
// Matrix operations
// =============================================================================

#define DEFINE_MATRIX_MULTIPLY(type, type_suffix) \
extern "C" __global__ void matrix_multiply_##type_suffix( \
    const type* a, const type* b, type* c, \
    unsigned int m, unsigned int k, unsigned int n) { \
    \
    int row = blockIdx.y * blockDim.y + threadIdx.y; \
    int col = blockIdx.x * blockDim.x + threadIdx.x; \
    \
    if (row < m && col < n) { \
        type sum = (type)0; \
        for (int i = 0; i < k; i++) { \
            sum += a[row * k + i] * b[i * n + col]; \
        } \
        c[row * n + col] = sum; \
    } \
}

// Optimized matrix multiply with shared memory
#define DEFINE_MATRIX_MULTIPLY_SHARED(type, type_suffix) \
extern "C" __global__ void matrix_multiply_shared_##type_suffix( \
    const type* a, const type* b, type* c, \
    unsigned int m, unsigned int k, unsigned int n) { \
    \
    __shared__ type As[16][16]; \
    __shared__ type Bs[16][16]; \
    \
    int bx = blockIdx.x, by = blockIdx.y; \
    int tx = threadIdx.x, ty = threadIdx.y; \
    int row = by * 16 + ty; \
    int col = bx * 16 + tx; \
    \
    type sum = (type)0; \
    \
    for (int tile = 0; tile < (k + 15) / 16; tile++) { \
        /* Load data into shared memory */ \
        if (row < m && tile * 16 + tx < k) \
            As[ty][tx] = a[row * k + tile * 16 + tx]; \
        else \
            As[ty][tx] = (type)0; \
        \
        if (col < n && tile * 16 + ty < k) \
            Bs[ty][tx] = b[(tile * 16 + ty) * n + col]; \
        else \
            Bs[ty][tx] = (type)0; \
        \
        __syncthreads(); \
        \
        /* Compute partial result */ \
        for (int i = 0; i < 16; i++) { \
            sum += As[ty][i] * Bs[i][tx]; \
        } \
        \
        __syncthreads(); \
    } \
    \
    if (row < m && col < n) { \
        c[row * n + col] = sum; \
    } \
}

// =============================================================================
// Transpose operations
// =============================================================================

#define DEFINE_TRANSPOSE(type, type_suffix) \
extern "C" __global__ void transpose_##type_suffix( \
    const type* input, type* output, \
    const unsigned int* input_dims, const unsigned int* input_strides, \
    const unsigned int* output_dims, const unsigned int* output_strides, \
    unsigned int ndim, unsigned int total_elements) { \
    \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < total_elements) { \
        unsigned int input_indices[8]; \
        unsigned int output_indices[8]; \
        \
        /* Unravel input index */ \
        unravel_index(idx, input_dims, ndim, input_indices); \
        \
        /* Reverse indices for transpose */ \
        for (int i = 0; i < ndim; i++) { \
            output_indices[i] = input_indices[ndim - 1 - i]; \
        } \
        \
        /* Compute output flat index */ \
        unsigned int output_idx = ravel_index(output_indices, output_strides, ndim); \
        \
        output[output_idx] = input[idx]; \
    } \
}

// 2D transpose with shared memory optimization
#define DEFINE_TRANSPOSE_2D_SHARED(type, type_suffix) \
extern "C" __global__ void transpose_2d_shared_##type_suffix( \
    const type* input, type* output, unsigned int rows, unsigned int cols) { \
    \
    __shared__ type tile[16][17]; /* +1 to avoid bank conflicts */ \
    \
    int x = blockIdx.x * 16 + threadIdx.x; \
    int y = blockIdx.y * 16 + threadIdx.y; \
    \
    /* Load data into shared memory */ \
    if (x < cols && y < rows) { \
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x]; \
    } \
    \
    __syncthreads(); \
    \
    /* Write transposed data */ \
    x = blockIdx.y * 16 + threadIdx.x; \
    y = blockIdx.x * 16 + threadIdx.y; \
    \
    if (x < rows && y < cols) { \
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y]; \
    } \
}

// =============================================================================
// Indexing and slicing operations
// =============================================================================

extern "C" __global__ void copy_element(
    const void* input, void* output, unsigned int index) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((float*)output)[0] = ((const float*)input)[index];
    }
}

extern "C" __global__ void set_element(
    void* array, unsigned int index, const void* value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((float*)array)[index] = *((const float*)value);
    }
}

extern "C" __global__ void slice_first_dim(
    const void* input, void* output,
    unsigned int start, unsigned int slice_len,
    unsigned int elements_per_slice, unsigned int total_output_elements) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_output_elements) {
        unsigned int slice_idx = idx / elements_per_slice;
        unsigned int element_idx = idx % elements_per_slice;
        unsigned int input_idx = (start + slice_idx) * elements_per_slice + element_idx;

        ((float*)output)[idx] = ((const float*)input)[input_idx];
    }
}

extern "C" __global__ void extract_column(
    const void* input, void* output,
    unsigned int rows, unsigned int cols, unsigned int col_index) {

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < rows) {
        ((float*)output)[row] = ((const float*)input)[row * cols + col_index];
    }
}

// =============================================================================
// Range and utility operations
// =============================================================================

#define DEFINE_RANGE_FILL(type, type_suffix) \
extern "C" __global__ void generic_range_##type_suffix( \
    type start, type step, unsigned int n, type* output) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < n) { \
        output[idx] = start + (type)idx * step; \
    } \
}

extern "C" __global__ void linspace_double(
    double start, double step, unsigned int n, double* output) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        output[idx] = start + (double)idx * step;
    }
}

extern "C" __global__ void copy_memory(
    const void* src, void* dst, unsigned int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        ((float*)dst)[idx] = ((const float*)src)[idx];
    }
}

extern "C" __global__ void fill_value(
    void* output, const void* value, unsigned int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        ((float*)output)[idx] = *((const float*)value);
    }
}

// =============================================================================
// Generic map, filter, reduce operations (placeholders)
// =============================================================================

extern "C" __global__ void generic_map(
    const void* input, void* output, unsigned int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        // Placeholder - just copy the data
        ((float*)output)[idx] = ((const float*)input)[idx];
    }
}

extern "C" __global__ void generic_filter(
    const void* input, void* output, unsigned int n, unsigned int* count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        // Placeholder filter - just copy all elements for now
        ((float*)output)[idx] = ((const float*)input)[idx];
        if (idx == 0) *count = n;
    }
}

extern "C" __global__ void generic_reduce(
    const void* input, unsigned int n, const void* initial, void* result) {
    // Placeholder - would need runtime compilation for arbitrary reduction functions
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *((float*)result) = *((const float*)initial);
    }
}

extern "C" __global__ void generic_search(
    const void* input, unsigned int n, int* result) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        // Placeholder search - return first index for now
        if (idx == 0) *result = 0;
    }
}

// =============================================================================
// Utility kernels
// =============================================================================

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

// =============================================================================
// Generate kernels for all supported types
// =============================================================================

// Basic element-wise operations
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

DEFINE_ELEMENTWISE_OP(add, +, short, short)
DEFINE_ELEMENTWISE_OP(sub, -, short, short)
DEFINE_ELEMENTWISE_OP(mul, *, short, short)
DEFINE_ELEMENTWISE_OP(div, /, short, short)

DEFINE_ELEMENTWISE_OP(add, +, unsigned short, ushort)
DEFINE_ELEMENTWISE_OP(sub, -, unsigned short, ushort)
DEFINE_ELEMENTWISE_OP(mul, *, unsigned short, ushort)
DEFINE_ELEMENTWISE_OP(div, /, unsigned short, ushort)

DEFINE_ELEMENTWISE_OP(add, +, char, char)
DEFINE_ELEMENTWISE_OP(sub, -, char, char)
DEFINE_ELEMENTWISE_OP(mul, *, char, char)
DEFINE_ELEMENTWISE_OP(div, /, char, char)

DEFINE_ELEMENTWISE_OP(add, +, unsigned char, uchar)
DEFINE_ELEMENTWISE_OP(sub, -, unsigned char, uchar)
DEFINE_ELEMENTWISE_OP(mul, *, unsigned char, uchar)
DEFINE_ELEMENTWISE_OP(div, /, unsigned char, uchar)

// Broadcasting operations
DEFINE_ELEMENTWISE_BROADCAST_OP(add, +, float, float)
DEFINE_ELEMENTWISE_BROADCAST_OP(sub, -, float, float)
DEFINE_ELEMENTWISE_BROADCAST_OP(mul, *, float, float)
DEFINE_ELEMENTWISE_BROADCAST_OP(div, /, float, float)

DEFINE_ELEMENTWISE_BROADCAST_OP(add, +, double, double)
DEFINE_ELEMENTWISE_BROADCAST_OP(sub, -, double, double)
DEFINE_ELEMENTWISE_BROADCAST_OP(mul, *, double, double)
DEFINE_ELEMENTWISE_BROADCAST_OP(div, /, double, double)

DEFINE_ELEMENTWISE_BROADCAST_OP(add, +, int, int)
DEFINE_ELEMENTWISE_BROADCAST_OP(sub, -, int, int)
DEFINE_ELEMENTWISE_BROADCAST_OP(mul, *, int, int)
DEFINE_ELEMENTWISE_BROADCAST_OP(div, /, int, int)

DEFINE_ELEMENTWISE_BROADCAST_OP(add, +, unsigned int, uint)
DEFINE_ELEMENTWISE_BROADCAST_OP(sub, -, unsigned int, uint)
DEFINE_ELEMENTWISE_BROADCAST_OP(mul, *, unsigned int, uint)
DEFINE_ELEMENTWISE_BROADCAST_OP(div, /, unsigned int, uint)

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

// Use atomicMax with type casting for floating point
DEFINE_REDUCE_MAX(float, float, atomicMax)
DEFINE_REDUCE_MAX(double, double, atomicMax)
DEFINE_REDUCE_MAX(int, int, atomicMax)
DEFINE_REDUCE_MAX(unsigned int, uint, atomicMax)

DEFINE_REDUCE_MIN(float, float, atomicMin)
DEFINE_REDUCE_MIN(double, double, atomicMin)
DEFINE_REDUCE_MIN(int, int, atomicMin)
DEFINE_REDUCE_MIN(unsigned int, uint, atomicMin)

// Axis reduction operations
DEFINE_REDUCE_SUM_AXIS(float, float)
DEFINE_REDUCE_SUM_AXIS(double, double)
DEFINE_REDUCE_SUM_AXIS(int, int)
DEFINE_REDUCE_SUM_AXIS(unsigned int, uint)

// Matrix operations
DEFINE_MATRIX_MULTIPLY(float, float)
DEFINE_MATRIX_MULTIPLY(double, double)
DEFINE_MATRIX_MULTIPLY(int, int)

DEFINE_MATRIX_MULTIPLY_SHARED(float, float)
DEFINE_MATRIX_MULTIPLY_SHARED(double, double)

// Transpose operations
DEFINE_TRANSPOSE(float, float)
DEFINE_TRANSPOSE(double, double)
DEFINE_TRANSPOSE(int, int)
DEFINE_TRANSPOSE(unsigned int, uint)
DEFINE_TRANSPOSE(long long, long)
DEFINE_TRANSPOSE(unsigned long long, ulong)

DEFINE_TRANSPOSE_2D_SHARED(float, float)
DEFINE_TRANSPOSE_2D_SHARED(double, double)

// Range operations
DEFINE_RANGE_FILL(float, float)
DEFINE_RANGE_FILL(double, double)
DEFINE_RANGE_FILL(int, int)
DEFINE_RANGE_FILL(unsigned int, uint)
DEFINE_RANGE_FILL(long long, long)
DEFINE_RANGE_FILL(unsigned long long, ulong)