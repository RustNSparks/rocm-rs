// src/rocblas/level3.rs

use crate::rocblas::ffi;
use crate::rocblas::handle::Handle;
use crate::rocblas::error::{Error, Result};
use crate::rocblas::types::{Operation, DataType};
use crate::rocblas::utils::GemmAlgo;

//==============================================================================
// GEMM functions - General Matrix-Matrix Multiplication
//==============================================================================

/// Matrix-matrix multiplication
/// 
/// Computes one of the following matrix-matrix operations:
/// 
/// C := alpha * A * B + beta * C
/// C := alpha * A^T * B + beta * C
/// C := alpha * A * B^T + beta * C
/// C := alpha * A^T * B^T + beta * C
/// 
/// where alpha and beta are scalars, and A, B, C are matrices.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `transa` - Operation op(A) that is non-or (conjugate) transpose
/// * `transb` - Operation op(B) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrix op(A) and C
/// * `n` - Number of columns of matrix op(B) and C
/// * `k` - Number of columns of matrix op(A) and rows of op(B)
/// * `alpha` - Scalar alpha
/// * `A` - Buffer storing matrix A
/// * `lda` - Leading dimension of matrix A
/// * `B` - Buffer storing matrix B
/// * `ldb` - Leading dimension of matrix B
/// * `beta` - Scalar beta
/// * `C` - Buffer storing matrix C
/// * `ldc` - Leading dimension of matrix C
pub fn gemm<T>(
    handle: &Handle,
    transa: Operation,
    transb: Operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    B: *const T,
    ldb: i32,
    beta: &T,
    C: *mut T,
    ldc: i32,
) -> Result<()>
where
    T: GemmType,
{
    T::rocblas_gemm(
        handle, transa, transb, m, n, k, 
        alpha, A, lda, B, ldb, beta, C, ldc,
    )
}

/// Batched matrix-matrix multiplication
/// 
/// Computes one of the following batched matrix-matrix operations:
/// 
/// C_i := alpha * op(A_i) * op(B_i) + beta * C_i
/// 
/// where (A_i, B_i, C_i) is the i-th instance of the batch.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `transa` - Operation op(A) that is non-or (conjugate) transpose
/// * `transb` - Operation op(B) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrix op(A_i) and C_i
/// * `n` - Number of columns of matrix op(B_i) and C_i
/// * `k` - Number of columns of matrix op(A_i) and rows of op(B_i)
/// * `alpha` - Scalar alpha
/// * `A` - Array of pointers to matrices A_i
/// * `lda` - Leading dimension of matrices A_i
/// * `B` - Array of pointers to matrices B_i
/// * `ldb` - Leading dimension of matrices B_i
/// * `beta` - Scalar beta
/// * `C` - Array of pointers to matrices C_i
/// * `ldc` - Leading dimension of matrices C_i
/// * `batch_count` - Number of instances in the batch
pub fn gemm_batched<T>(
    handle: &Handle,
    transa: Operation,
    transb: Operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const *const T,
    lda: i32,
    B: *const *const T,
    ldb: i32,
    beta: &T,
    C: *const *mut T,
    ldc: i32,
    batch_count: i32,
) -> Result<()>
where
    T: GemmBatchedType,
{
    T::rocblas_gemm_batched(
        handle, transa, transb, m, n, k, 
        alpha, A, lda, B, ldb, beta, C, ldc, batch_count,
    )
}

/// Strided batched matrix-matrix multiplication
/// 
/// Computes one of the following strided batched matrix-matrix operations:
/// 
/// C_i := alpha * op(A_i) * op(B_i) + beta * C_i
/// 
/// where (A_i, B_i, C_i) is the i-th instance of the batch.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `transa` - Operation op(A) that is non-or (conjugate) transpose
/// * `transb` - Operation op(B) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrix op(A_i) and C_i
/// * `n` - Number of columns of matrix op(B_i) and C_i
/// * `k` - Number of columns of matrix op(A_i) and rows of op(B_i)
/// * `alpha` - Scalar alpha
/// * `A` - Pointer to the first matrix A_1
/// * `lda` - Leading dimension of matrices A_i
/// * `stride_A` - Stride from start of one matrix (A_i) to the next (A_i+1)
/// * `B` - Pointer to the first matrix B_1
/// * `ldb` - Leading dimension of matrices B_i
/// * `stride_B` - Stride from start of one matrix (B_i) to the next (B_i+1)
/// * `beta` - Scalar beta
/// * `C` - Pointer to the first matrix C_1
/// * `ldc` - Leading dimension of matrices C_i
/// * `stride_C` - Stride from start of one matrix (C_i) to the next (C_i+1)
/// * `batch_count` - Number of instances in the batch
pub fn gemm_strided_batched<T>(
    handle: &Handle,
    transa: Operation,
    transb: Operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    stride_A: i64,
    B: *const T,
    ldb: i32,
    stride_B: i64,
    beta: &T,
    C: *mut T,
    ldc: i32,
    stride_C: i64,
    batch_count: i32,
) -> Result<()>
where
    T: GemmStridedBatchedType,
{
    T::rocblas_gemm_strided_batched(
        handle, transa, transb, m, n, k, 
        alpha, A, lda, stride_A, B, ldb, stride_B, 
        beta, C, ldc, stride_C, batch_count,
    )
}

/// General matrix-matrix multiplication with extended precision
///
/// Computes the general matrix-matrix product with extended precision
/// where the data types of matrices can be different.
///
/// C := alpha * op(A) * op(B) + beta * C
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `transa` - Operation op(A) that is non-or (conjugate) transpose
/// * `transb` - Operation op(B) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrix op(A) and C
/// * `n` - Number of columns of matrix op(B) and C
/// * `k` - Number of columns of matrix op(A) and rows of op(B)
/// * `alpha` - Scalar alpha
/// * `A` - Buffer storing matrix A
/// * `a_type` - Data type of matrix A
/// * `lda` - Leading dimension of matrix A
/// * `B` - Buffer storing matrix B
/// * `b_type` - Data type of matrix B
/// * `ldb` - Leading dimension of matrix B
/// * `beta` - Scalar beta
/// * `C` - Buffer storing matrix C
/// * `c_type` - Data type of matrix C
/// * `ldc` - Leading dimension of matrix C
/// * `compute_type` - Computation type
/// * `algo` - GEMM algorithm
pub fn gemm_ex(
    handle: &Handle,
    transa: Operation,
    transb: Operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const std::ffi::c_void,
    A: *const std::ffi::c_void,
    a_type: DataType,
    lda: i32,
    B: *const std::ffi::c_void,
    b_type: DataType,
    ldb: i32,
    beta: *const std::ffi::c_void,
    C: *mut std::ffi::c_void,
    c_type: DataType,
    ldc: i32,
    compute_type: DataType,
    algo: GemmAlgo,
) -> Result<()> {
    let status = unsafe {
        ffi::rocblas_gemm_ex(
            handle.as_raw(),
            transa.into(),
            transb.into(),
            m,
            n,
            k,
            alpha,
            A,
            a_type.into(),
            lda,
            B,
            b_type.into(),
            ldb,
            beta,
            C,
            c_type.into(),
            ldc,
            C, // Output matrix same as C
            c_type.into(),
            ldc,
            compute_type.into(),
            algo.into(),
            0, // Solution index (0 means auto)
            0, // Flags
        )
    };
    
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    
    Ok(())
}

//==============================================================================
// Type traits for implementation
//==============================================================================

/// Trait for types that can be used with gemm
pub trait GemmType {
    fn rocblas_gemm(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()>;
}

impl GemmType for f32 {
    fn rocblas_gemm(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sgemm(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmType for f64 {
    fn rocblas_gemm(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dgemm(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmType for ffi::rocblas_float_complex {
    fn rocblas_gemm(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgemm(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmType for ffi::rocblas_double_complex {
    fn rocblas_gemm(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgemm(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with gemm_batched
pub trait GemmBatchedType {
    fn rocblas_gemm_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl GemmBatchedType for f32 {
    fn rocblas_gemm_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sgemm_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmBatchedType for f64 {
    fn rocblas_gemm_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dgemm_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmBatchedType for ffi::rocblas_float_complex {
    fn rocblas_gemm_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgemm_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmBatchedType for ffi::rocblas_double_complex {
    fn rocblas_gemm_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgemm_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with gemm_strided_batched
pub trait GemmStridedBatchedType {
    fn rocblas_gemm_strided_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl GemmStridedBatchedType for f32 {
    fn rocblas_gemm_strided_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sgemm_strided_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                B,
                ldb,
                stride_B,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmStridedBatchedType for f64 {
    fn rocblas_gemm_strided_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dgemm_strided_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                B,
                ldb,
                stride_B,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmStridedBatchedType for ffi::rocblas_float_complex {
    fn rocblas_gemm_strided_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgemm_strided_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                B,
                ldb,
                stride_B,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmStridedBatchedType for ffi::rocblas_double_complex {
    fn rocblas_gemm_strided_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgemm_strided_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                B,
                ldb,
                stride_B,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}