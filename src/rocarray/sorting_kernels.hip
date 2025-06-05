// src/rocarray/sorting_kernels.hip
#include <hip/hip_runtime.h>

// Bitonic sort for small arrays (power of 2 sizes up to 1024)
#define DEFINE_BITONIC_SORT(type, type_suffix, ascending) \
extern "C" __global__ void bitonic_sort_##ascending##_##type_suffix( \
    type* data, unsigned int n, unsigned int padded_n) { \
    int tid = threadIdx.x; \
    int bid = blockIdx.x; \
    int idx = bid * blockDim.x + tid; \
    \
    __shared__ type shared_data[1024]; \
    \
    /* Load data into shared memory */ \
    if (idx < n) { \
        shared_data[tid] = data[idx]; \
    } else if (idx < padded_n) { \
        shared_data[tid] = ascending ? (type)INFINITY : (type)-INFINITY; \
    } \
    __syncthreads(); \
    \
    /* Bitonic sort */ \
    for (int size = 2; size <= blockDim.x; size <<= 1) { \
        for (int stride = size >> 1; stride > 0; stride >>= 1) { \
            int pos = 2 * tid - (tid & (stride - 1)); \
            bool should_swap; \
            if (ascending) { \
                should_swap = shared_data[pos] > shared_data[pos + stride]; \
            } else { \
                should_swap = shared_data[pos] < shared_data[pos + stride]; \
            } \
            \
            if (should_swap) { \
                type temp = shared_data[pos]; \
                shared_data[pos] = shared_data[pos + stride]; \
                shared_data[pos + stride] = temp; \
            } \
            __syncthreads(); \
        } \
    } \
    \
    /* Write back to global memory */ \
    if (idx < n) { \
        data[idx] = shared_data[tid]; \
    } \
}

// Radix sort implementation (simplified)
#define DEFINE_RADIX_SORT(type, type_suffix) \
extern "C" __global__ void radix_sort_ascending_##type_suffix( \
    type* data, type* temp_buffer, unsigned int n) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    \
    /* This is a simplified radix sort - real implementation would be more complex */ \
    /* For now, we'll use a simple bubble sort approach for demonstration */ \
    if (idx == 0) { \
        for (int i = 0; i < n - 1; i++) { \
            for (int j = 0; j < n - i - 1; j++) { \
                if (data[j] > data[j + 1]) { \
                    type temp = data[j]; \
                    data[j] = data[j + 1]; \
                    data[j + 1] = temp; \
                } \
            } \
        } \
    } \
}

// Initialize indices for argsort
extern "C" __global__ void init_indices(unsigned int* indices, unsigned int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        indices[idx] = idx;
    }
}

// Argsort implementation
#define DEFINE_ARGSORT(type, type_suffix) \
extern "C" __global__ void argsort_##type_suffix( \
    const type* data, unsigned int* indices, unsigned int n) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    \
    /* Simple bubble sort on indices based on data values */ \
    if (idx == 0) { \
        for (int i = 0; i < n - 1; i++) { \
            for (int j = 0; j < n - i - 1; j++) { \
                if (data[indices[j]] > data[indices[j + 1]]) { \
                    unsigned int temp = indices[j]; \
                    indices[j] = indices[j + 1]; \
                    indices[j + 1] = temp; \
                } \
            } \
        } \
    } \
}

// Check if array is sorted
#define DEFINE_IS_SORTED(type, type_suffix) \
extern "C" __global__ void is_sorted_##type_suffix( \
    const type* data, unsigned int n, unsigned int* result) { \
    __shared__ unsigned int block_result[256]; \
    int tid = threadIdx.x; \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    \
    block_result[tid] = 1; /* Assume sorted */ \
    \
    if (idx < n - 1) { \
        if (data[idx] > data[idx + 1]) { \
            block_result[tid] = 0; /* Not sorted */ \
        } \
    } \
    \
    __syncthreads(); \
    \
    /* Reduce within block */ \
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { \
        if (tid < s) { \
            block_result[tid] = block_result[tid] && block_result[tid + s]; \
        } \
        __syncthreads(); \
    } \
    \
    if (tid == 0) { \
        atomicAnd(result, block_result[0]); \
    } \
}

// Partial sort (sort first k elements)
#define DEFINE_PARTIAL_SORT(type, type_suffix) \
extern "C" __global__ void partial_sort_##type_suffix( \
    type* data, unsigned int n, unsigned int k) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    \
    /* Simple selection sort for first k elements */ \
    if (idx == 0) { \
        for (int i = 0; i < k && i < n; i++) { \
            int min_idx = i; \
            for (int j = i + 1; j < n; j++) { \
                if (data[j] < data[min_idx]) { \
                    min_idx = j; \
                } \
            } \
            if (min_idx != i) { \
                type temp = data[i]; \
                data[i] = data[min_idx]; \
                data[min_idx] = temp; \
            } \
        } \
    } \
}

// Nth element (quickselect)
#define DEFINE_NTH_ELEMENT(type, type_suffix) \
extern "C" __global__ void nth_element_##type_suffix( \
    type* data, unsigned int n, unsigned int nth, type* result) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    \
    if (idx == 0) { \
        /* Simple approach: sort and take nth element */ \
        for (int i = 0; i < n - 1; i++) { \
            for (int j = 0; j < n - i - 1; j++) { \
                if (data[j] > data[j + 1]) { \
                    type temp = data[j]; \
                    data[j] = data[j + 1]; \
                    data[j + 1] = temp; \
                } \
            } \
        } \
        if (nth < n) { \
            *result = data[nth]; \
        } \
    } \
}

// Merge two sorted arrays
#define DEFINE_MERGE_SORTED(type, type_suffix) \
extern "C" __global__ void merge_sorted_##type_suffix( \
    const type* left, unsigned int left_len, \
    const type* right, unsigned int right_len, \
    type* output) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    int total_len = left_len + right_len; \
    \
    if (idx < total_len) { \
        /* Binary search approach for parallel merge */ \
        int left_idx = 0; \
        int right_idx = 0; \
        \
        /* Find position in merged array */ \
        for (int i = 0; i < idx; i++) { \
            if (left_idx >= left_len) { \
                right_idx++; \
            } else if (right_idx >= right_len) { \
                left_idx++; \
            } else if (left[left_idx] <= right[right_idx]) { \
                left_idx++; \
            } else { \
                right_idx++; \
            } \
        } \
        \
        /* Assign value */ \
        if (left_idx >= left_len) { \
            output[idx] = right[right_idx]; \
        } else if (right_idx >= right_len) { \
            output[idx] = left[left_idx]; \
        } else if (left[left_idx] <= right[right_idx]) { \
            output[idx] = left[left_idx]; \
        } else { \
            output[idx] = right[right_idx]; \
        } \
    } \
}

// Stable sort (merge sort)
#define DEFINE_STABLE_SORT(type, type_suffix) \
extern "C" __global__ void stable_sort_##type_suffix( \
    type* data, type* temp_buffer, unsigned int n) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    \
    /* Simplified stable sort - real implementation would use proper merge sort */ \
    if (idx == 0) { \
        /* Copy to temp buffer */ \
        for (int i = 0; i < n; i++) { \
            temp_buffer[i] = data[i]; \
        } \
        \
        /* Simple stable sort (insertion sort) */ \
        for (int i = 1; i < n; i++) { \
            type key = temp_buffer[i]; \
            int j = i - 1; \
            while (j >= 0 && temp_buffer[j] > key) { \
                temp_buffer[j + 1] = temp_buffer[j]; \
                j--; \
            } \
            temp_buffer[j + 1] = key; \
        } \
        \
        /* Copy back */ \
        for (int i = 0; i < n; i++) { \
            data[i] = temp_buffer[i]; \
        } \
    } \
}

// Generate kernels for all supported types
DEFINE_BITONIC_SORT(float, float, ascending)
DEFINE_BITONIC_SORT(float, float, descending)
DEFINE_BITONIC_SORT(double, double, ascending)
DEFINE_BITONIC_SORT(double, double, descending)
DEFINE_BITONIC_SORT(int, int, ascending)
DEFINE_BITONIC_SORT(int, int, descending)
DEFINE_BITONIC_SORT(unsigned int, uint, ascending)
DEFINE_BITONIC_SORT(unsigned int, uint, descending)

DEFINE_RADIX_SORT(float, float)
DEFINE_RADIX_SORT(double, double)
DEFINE_RADIX_SORT(int, int)
DEFINE_RADIX_SORT(unsigned int, uint)
DEFINE_RADIX_SORT(long long, long)
DEFINE_RADIX_SORT(unsigned long long, ulong)

DEFINE_ARGSORT(float, float)
DEFINE_ARGSORT(double, double)
DEFINE_ARGSORT(int, int)
DEFINE_ARGSORT(unsigned int, uint)
DEFINE_ARGSORT(long long, long)
DEFINE_ARGSORT(unsigned long long, ulong)

DEFINE_IS_SORTED(float, float)
DEFINE_IS_SORTED(double, double)
DEFINE_IS_SORTED(int, int)
DEFINE_IS_SORTED(unsigned int, uint)
DEFINE_IS_SORTED(long long, long)
DEFINE_IS_SORTED(unsigned long long, ulong)

DEFINE_PARTIAL_SORT(float, float)
DEFINE_PARTIAL_SORT(double, double)
DEFINE_PARTIAL_SORT(int, int)
DEFINE_PARTIAL_SORT(unsigned int, uint)

DEFINE_NTH_ELEMENT(float, float)
DEFINE_NTH_ELEMENT(double, double)
DEFINE_NTH_ELEMENT(int, int)
DEFINE_NTH_ELEMENT(unsigned int, uint)

DEFINE_MERGE_SORTED(float, float)
DEFINE_MERGE_SORTED(double, double)
DEFINE_MERGE_SORTED(int, int)
DEFINE_MERGE_SORTED(unsigned int, uint)

DEFINE_STABLE_SORT(float, float)
DEFINE_STABLE_SORT(double, double)
DEFINE_STABLE_SORT(int, int)
DEFINE_STABLE_SORT(unsigned int, uint)