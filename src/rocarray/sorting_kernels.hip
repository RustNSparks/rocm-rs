// src/rocarray/sorting_kernels.hip
#include <hip/hip_runtime.h>

#define ASCENDING 1
#define DESCENDING 0

// Simple bubble sort kernel for debugging
extern "C" __global__ void simple_sort_int(int* data, unsigned int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Only use thread 0 for simplicity
    if (tid == 0 && bid == 0) {
        // Simple bubble sort
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (data[j] > data[j + 1]) {
                    int temp = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = temp;
                }
            }
        }
    }
}

// Bitonic sort with fixed shared memory
#define DEFINE_BITONIC_SORT(type, type_suffix, is_ascending) \
extern "C" __global__ void bitonic_sort_##is_ascending##_##type_suffix( \
    type* data, unsigned int n, unsigned int padded_n) { \
    int tid = threadIdx.x; \
    int bid = blockIdx.x; \
    int idx = bid * blockDim.x + tid; \
    \
    __shared__ type sdata[512]; /* Fixed smaller size */ \
    \
    /* Bounds check */ \
    if (tid >= 512 || blockDim.x > 512) return; \
    \
    /* Load data into shared memory */ \
    if (idx < n) { \
        sdata[tid] = data[idx]; \
    } else { \
        /* Use simple sentinel values */ \
        if (is_ascending) { \
            sdata[tid] = (type)999999; /* Large value for ascending */ \
        } else { \
            sdata[tid] = (type)-999999; /* Small value for descending */ \
        } \
    } \
    __syncthreads(); \
    \
    /* Bitonic sort */ \
    for (int size = 2; size <= blockDim.x; size <<= 1) { \
        for (int stride = size >> 1; stride > 0; stride >>= 1) { \
            int partner = tid ^ stride; \
            if (partner < blockDim.x && partner < 512) { \
                bool ascending_block = ((tid & size) == 0) == is_ascending; \
                bool should_swap = ascending_block ? \
                    (sdata[tid] > sdata[partner]) : \
                    (sdata[tid] < sdata[partner]); \
                \
                if (should_swap) { \
                    type temp = sdata[tid]; \
                    sdata[tid] = sdata[partner]; \
                    sdata[partner] = temp; \
                } \
            } \
            __syncthreads(); \
        } \
    } \
    \
    /* Write back to global memory */ \
    if (idx < n) { \
        data[idx] = sdata[tid]; \
    } \
}

// Simple radix sort fallback
#define DEFINE_SIMPLE_RADIX_SORT(type, type_suffix) \
extern "C" __global__ void radix_sort_ascending_##type_suffix( \
    type* data, type* temp_buffer, unsigned int n) { \
    int tid = threadIdx.x; \
    int bid = blockIdx.x; \
    \
    /* Use simple bubble sort for now */ \
    if (tid == 0 && bid == 0) { \
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

// Simple argsort
#define DEFINE_ARGSORT(type, type_suffix) \
extern "C" __global__ void argsort_##type_suffix( \
    const type* data, unsigned int* indices, unsigned int n) { \
    int tid = threadIdx.x; \
    int bid = blockIdx.x; \
    \
    /* Use simple bubble sort on indices */ \
    if (tid == 0 && bid == 0) { \
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
    int tid = threadIdx.x; \
    int bid = blockIdx.x; \
    \
    if (tid == 0 && bid == 0) { \
        *result = 1; /* Assume sorted */ \
        for (int i = 0; i < n - 1; i++) { \
            if (data[i] > data[i + 1]) { \
                *result = 0; \
                break; \
            } \
        } \
    } \
}

// Simple partial sort
#define DEFINE_PARTIAL_SORT(type, type_suffix) \
extern "C" __global__ void partial_sort_##type_suffix( \
    type* data, unsigned int n, unsigned int k) { \
    int tid = threadIdx.x; \
    int bid = blockIdx.x; \
    \
    if (tid == 0 && bid == 0) { \
        /* Selection sort for first k elements */ \
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

// Simple nth element
#define DEFINE_NTH_ELEMENT(type, type_suffix) \
extern "C" __global__ void nth_element_##type_suffix( \
    type* data, unsigned int n, unsigned int nth, type* result) { \
    int tid = threadIdx.x; \
    int bid = blockIdx.x; \
    \
    if (tid == 0 && bid == 0 && nth < n) { \
        /* Simple approach: sort and take nth */ \
        for (int i = 0; i < n - 1; i++) { \
            for (int j = 0; j < n - i - 1; j++) { \
                if (data[j] > data[j + 1]) { \
                    type temp = data[j]; \
                    data[j] = data[j + 1]; \
                    data[j + 1] = temp; \
                } \
            } \
        } \
        *result = data[nth]; \
    } \
}

// Simple merge
#define DEFINE_MERGE_SORTED(type, type_suffix) \
extern "C" __global__ void merge_sorted_##type_suffix( \
    const type* left, unsigned int left_len, \
    const type* right, unsigned int right_len, \
    type* output) { \
    int tid = threadIdx.x; \
    int bid = blockIdx.x; \
    \
    if (tid == 0 && bid == 0) { \
        int i = 0, j = 0, k = 0; \
        while (i < left_len && j < right_len) { \
            if (left[i] <= right[j]) { \
                output[k++] = left[i++]; \
            } else { \
                output[k++] = right[j++]; \
            } \
        } \
        while (i < left_len) { \
            output[k++] = left[i++]; \
        } \
        while (j < right_len) { \
            output[k++] = right[j++]; \
        } \
    } \
}

// Simple stable sort
#define DEFINE_STABLE_SORT(type, type_suffix) \
extern "C" __global__ void stable_sort_##type_suffix( \
    type* data, type* temp_buffer, unsigned int n) { \
    int tid = threadIdx.x; \
    int bid = blockIdx.x; \
    \
    if (tid == 0 && bid == 0) { \
        /* Insertion sort (stable) */ \
        for (int i = 1; i < n; i++) { \
            type key = data[i]; \
            int j = i - 1; \
            while (j >= 0 && data[j] > key) { \
                data[j + 1] = data[j]; \
                j--; \
            } \
            data[j + 1] = key; \
        } \
    } \
}

// Generate kernels for all supported types
DEFINE_BITONIC_SORT(float, float, ASCENDING)
DEFINE_BITONIC_SORT(float, float, DESCENDING)
DEFINE_BITONIC_SORT(double, double, ASCENDING)
DEFINE_BITONIC_SORT(double, double, DESCENDING)
DEFINE_BITONIC_SORT(int, int, ASCENDING)
DEFINE_BITONIC_SORT(int, int, DESCENDING)
DEFINE_BITONIC_SORT(unsigned int, uint, ASCENDING)
DEFINE_BITONIC_SORT(unsigned int, uint, DESCENDING)

DEFINE_SIMPLE_RADIX_SORT(float, float)
DEFINE_SIMPLE_RADIX_SORT(double, double)
DEFINE_SIMPLE_RADIX_SORT(int, int)
DEFINE_SIMPLE_RADIX_SORT(unsigned int, uint)
DEFINE_SIMPLE_RADIX_SORT(long long, long)
DEFINE_SIMPLE_RADIX_SORT(unsigned long long, ulong)

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


// Add this simple test kernel to your sorting_kernels.hip
extern "C" __global__ void test_simple_kernel(int* data, unsigned int n) {
    // Do absolutely nothing - just test if kernel can be called
    int tid = threadIdx.x;
    if (tid == 0) {
        // Just touch the first element
        if (n > 0) {
            data[0] = data[0];
        }
    }
}