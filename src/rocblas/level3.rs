// src/rocblas/level3.rs

use crate::rocblas::ffi;
use crate::rocblas::handle::Handle;
use crate::rocblas::error::{Error, Result};
use crate::rocblas::types::{Operation, DataType};
use crate::rocblas::utils::GemmAlgo;

use super::types::{Fill, Side};

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

// Trait definitions for HEMM operations
pub trait HemmType {
    fn rocblas_hemm(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
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

impl HemmType for ffi::rocblas_float_complex {
    fn rocblas_hemm(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
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
            ffi::rocblas_chemm(
                handle.as_raw(),
                side.into(),
                uplo.into(),
                m,
                n,
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

impl HemmType for ffi::rocblas_double_complex {
    fn rocblas_hemm(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
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
            ffi::rocblas_zhemm(
                handle.as_raw(),
                side.into(),
                uplo.into(),
                m,
                n,
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

// Trait for HERK operations
pub trait HerkType {
    type ScalarType;
    
    fn rocblas_herk(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const Self,
        lda: i32,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()>;
}

impl HerkType for ffi::rocblas_float_complex {
    type ScalarType = f32;
    
    fn rocblas_herk(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const Self,
        lda: i32,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cherk(
                handle.as_raw(),
                uplo.into(),
                transA.into(),
                n,
                k,
                alpha,
                A,
                lda,
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

impl HerkType for ffi::rocblas_double_complex {
    type ScalarType = f64;
    
    fn rocblas_herk(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const Self,
        lda: i32,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zherk(
                handle.as_raw(),
                uplo.into(),
                transA.into(),
                n,
                k,
                alpha,
                A,
                lda,
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

// Trait definitions for SPR operations (packed symmetric rank-1 update)
pub trait SprType {
    fn rocblas_spr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        AP: *mut Self,
    ) -> Result<()>;
}

impl SprType for f32 {
    fn rocblas_spr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        AP: *mut Self,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sspr(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                AP,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl SprType for f64 {
    fn rocblas_spr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        AP: *mut Self,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dspr(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                AP,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// There are also complex versions in the bindings
impl SprType for ffi::rocblas_float_complex {
    fn rocblas_spr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        AP: *mut Self,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cspr(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                AP,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl SprType for ffi::rocblas_double_complex {
    fn rocblas_spr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        AP: *mut Self,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zspr(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                AP,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Trait for SPR2 operations (packed symmetric rank-2 update)
pub trait Spr2Type {
    fn rocblas_spr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        AP: *mut Self,
    ) -> Result<()>;
}

impl Spr2Type for f32 {
    fn rocblas_spr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        AP: *mut Self,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sspr2(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                AP,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl Spr2Type for f64 {
    fn rocblas_spr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        AP: *mut Self,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dspr2(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                AP,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Trait for SYR operations (symmetric rank-1 update)
pub trait SyrType {
    fn rocblas_syr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()>;
}

impl SyrType for f32 {
    fn rocblas_syr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ssyr(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                A,
                lda,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Trait for SYR2 operations (symmetric rank-2 update)
pub trait Syr2Type {
    fn rocblas_syr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()>;
}

impl Syr2Type for f32 {
    fn rocblas_syr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ssyr2(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                A,
                lda,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl Syr2Type for f64 {
    fn rocblas_syr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dsyr2(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                A,
                lda,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Implementations for complex versions (CSYR, ZSYR, CSYR2, ZSYR2)
impl SyrType for ffi::rocblas_float_complex {
    fn rocblas_syr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_csyr(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                A,
                lda,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl SyrType for ffi::rocblas_double_complex {
    fn rocblas_syr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zsyr(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                A,
                lda,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl Syr2Type for ffi::rocblas_float_complex {
    fn rocblas_syr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_csyr2(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                A,
                lda,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl Syr2Type for ffi::rocblas_double_complex {
    fn rocblas_syr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zsyr2(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                A,
                lda,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Batched and strided batched implementations for SYR and SYR2

pub trait SyrBatchedType {
    fn rocblas_syr_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl SyrBatchedType for f32 {
    fn rocblas_syr_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ssyr_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                A,
                lda,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Similar implementations for other data types and strided batched versions

pub trait SyrStridedBatchedType {
    fn rocblas_syr_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl SyrStridedBatchedType for f32 {
    fn rocblas_syr_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ssyr_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                stride_x,
                A,
                lda,
                stride_A,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Additional implementations for batched versions of HEMM and HERK

pub trait HemmBatchedType {
    fn rocblas_hemm_batched(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
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

impl HemmBatchedType for ffi::rocblas_float_complex {
    fn rocblas_hemm_batched(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
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
            ffi::rocblas_chemm_batched(
                handle.as_raw(),
                side.into(),
                uplo.into(),
                m,
                n,
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

impl HemmBatchedType for ffi::rocblas_double_complex {
    fn rocblas_hemm_batched(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
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
            ffi::rocblas_zhemm_batched(
                handle.as_raw(),
                side.into(),
                uplo.into(),
                m,
                n,
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

pub trait HemmStridedBatchedType {
    fn rocblas_hemm_strided_batched(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
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

impl HemmStridedBatchedType for ffi::rocblas_float_complex {
    fn rocblas_hemm_strided_batched(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
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
            ffi::rocblas_chemm_strided_batched(
                handle.as_raw(),
                side.into(),
                uplo.into(),
                m,
                n,
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

impl HemmStridedBatchedType for ffi::rocblas_double_complex {
    fn rocblas_hemm_strided_batched(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
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
            ffi::rocblas_zhemm_strided_batched(
                handle.as_raw(),
                side.into(),
                uplo.into(),
                m,
                n,
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

pub trait HerkBatchedType {
    type ScalarType;
    
    fn rocblas_herk_batched(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const *const Self,
        lda: i32,
        beta: &Self::ScalarType,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl HerkBatchedType for ffi::rocblas_float_complex {
    type ScalarType = f32;
    
    fn rocblas_herk_batched(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const *const Self,
        lda: i32,
        beta: &Self::ScalarType,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cherk_batched(
                handle.as_raw(),
                uplo.into(),
                transA.into(),
                n,
                k,
                alpha,
                A,
                lda,
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

impl HerkBatchedType for ffi::rocblas_double_complex {
    type ScalarType = f64;
    
    fn rocblas_herk_batched(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const *const Self,
        lda: i32,
        beta: &Self::ScalarType,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zherk_batched(
                handle.as_raw(),
                uplo.into(),
                transA.into(),
                n,
                k,
                alpha,
                A,
                lda,
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

pub trait HerkStridedBatchedType {
    type ScalarType;
    
    fn rocblas_herk_strided_batched(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl HerkStridedBatchedType for ffi::rocblas_float_complex {
    type ScalarType = f32;
    
    fn rocblas_herk_strided_batched(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cherk_strided_batched(
                handle.as_raw(),
                uplo.into(),
                transA.into(),
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
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

impl HerkStridedBatchedType for ffi::rocblas_double_complex {
    type ScalarType = f64;
    
    fn rocblas_herk_strided_batched(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zherk_strided_batched(
                handle.as_raw(),
                uplo.into(),
                transA.into(),
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
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

pub fn hemm_batched<T>(
    handle: &Handle,
    side: Side,
    uplo: Fill,
    m: i32,
    n: i32,
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
    T: HemmBatchedType,
{
    T::rocblas_hemm_batched(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batch_count)
}

/// Strided batched Hermitian matrix-matrix multiplication
pub fn hemm_strided_batched<T>(
    handle: &Handle,
    side: Side,
    uplo: Fill,
    m: i32,
    n: i32,
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
    T: HemmStridedBatchedType,
{
    T::rocblas_hemm_strided_batched(
        handle, side, uplo, m, n, alpha, A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count,
    )
}

/// Batched Hermitian rank-k update
pub fn herk_batched<T, R>(
    handle: &Handle,
    uplo: Fill,
    transA: Operation,
    n: i32,
    k: i32,
    alpha: &R,
    A: *const *const T,
    lda: i32,
    beta: &R,
    C: *const *mut T,
    ldc: i32,
    batch_count: i32,
) -> Result<()>
where
    T: HerkBatchedType<ScalarType = R>,
{
    T::rocblas_herk_batched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batch_count)
}

/// Strided batched Hermitian rank-k update
pub fn herk_strided_batched<T, R>(
    handle: &Handle,
    uplo: Fill,
    transA: Operation,
    n: i32,
    k: i32,
    alpha: &R,
    A: *const T,
    lda: i32,
    stride_A: i64,
    beta: &R,
    C: *mut T,
    ldc: i32,
    stride_C: i64,
    batch_count: i32,
) -> Result<()>
where
    T: HerkStridedBatchedType<ScalarType = R>,
{
    T::rocblas_herk_strided_batched(
        handle, uplo, transA, n, k, alpha, A, lda, stride_A, beta, C, ldc, stride_C, batch_count,
    )
}


/// Hermitian rank-k update with two matrices
/// 
/// Computes the matrix-matrix operation:
/// 
/// C := alpha * op(A) * op(B)^H + beta * C
/// 
/// This routine should only be used when the result of op(A)*op(B)^H will be Hermitian.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part of C is used
/// * `trans` - Operation op(A) that is non-or-conjugate transpose
/// * `n` - Number of rows and columns of matrix C
/// * `k` - Number of columns of op(A) and op(B)
/// * `alpha` - Scalar alpha
/// * `A` - Buffer storing matrix A
/// * `lda` - Leading dimension of matrix A
/// * `B` - Buffer storing matrix B
/// * `ldb` - Leading dimension of matrix B
/// * `beta` - Scalar beta
/// * `C` - Buffer storing matrix C
/// * `ldc` - Leading dimension of matrix C
pub fn herkx<T, R>(
    handle: &Handle,
    uplo: Fill,
    trans: Operation,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    B: *const T,
    ldb: i32,
    beta: &R,
    C: *mut T,
    ldc: i32,
) -> Result<()>
where
    T: HerkxType<ScalarType = R>,
{
    T::rocblas_herkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
}

/// Batched Hermitian rank-k update with two matrices
pub fn herkx_batched<T, R>(
    handle: &Handle,
    uplo: Fill,
    trans: Operation,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const *const T,
    lda: i32,
    B: *const *const T,
    ldb: i32,
    beta: &R,
    C: *const *mut T,
    ldc: i32,
    batch_count: i32,
) -> Result<()>
where
    T: HerkxBatchedType<ScalarType = R>,
{
    T::rocblas_herkx_batched(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count)
}

/// Strided batched Hermitian rank-k update with two matrices
pub fn herkx_strided_batched<T, R>(
    handle: &Handle,
    uplo: Fill,
    trans: Operation,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    stride_A: i64,
    B: *const T,
    ldb: i32,
    stride_B: i64,
    beta: &R,
    C: *mut T,
    ldc: i32,
    stride_C: i64,
    batch_count: i32,
) -> Result<()>
where
    T: HerkxStridedBatchedType<ScalarType = R>,
{
    T::rocblas_herkx_strided_batched(
        handle, uplo, trans, n, k, alpha, A, lda, stride_A, 
        B, ldb, stride_B, beta, C, ldc, stride_C, batch_count,
    )
}

/// Trait for types that can be used with herkx
pub trait HerkxType {
    type ScalarType;
    
    fn rocblas_herkx(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()>;
}

impl HerkxType for ffi::rocblas_float_complex {
    type ScalarType = f32;
    
    fn rocblas_herkx(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cherkx(
                handle.as_raw(),
                uplo.into(),
                trans.into(),
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

impl HerkxType for ffi::rocblas_double_complex {
    type ScalarType = f64;
    
    fn rocblas_herkx(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zherkx(
                handle.as_raw(),
                uplo.into(),
                trans.into(),
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

/// Trait for types that can be used with herkx_batched
pub trait HerkxBatchedType {
    type ScalarType;
    
    fn rocblas_herkx_batched(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self::ScalarType,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl HerkxBatchedType for ffi::rocblas_float_complex {
    type ScalarType = f32;
    
    fn rocblas_herkx_batched(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self::ScalarType,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cherkx_batched(
                handle.as_raw(),
                uplo.into(),
                trans.into(),
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

impl HerkxBatchedType for ffi::rocblas_double_complex {
    type ScalarType = f64;
    
    fn rocblas_herkx_batched(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self::ScalarType,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zherkx_batched(
                handle.as_raw(),
                uplo.into(),
                trans.into(),
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

/// Trait for types that can be used with herkx_strided_batched
pub trait HerkxStridedBatchedType {
    type ScalarType;
    
    fn rocblas_herkx_strided_batched(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl HerkxStridedBatchedType for ffi::rocblas_float_complex {
    type ScalarType = f32;
    
    fn rocblas_herkx_strided_batched(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cherkx_strided_batched(
                handle.as_raw(),
                uplo.into(),
                trans.into(),
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

impl HerkxStridedBatchedType for ffi::rocblas_double_complex {
    type ScalarType = f64;
    
    fn rocblas_herkx_strided_batched(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zherkx_strided_batched(
                handle.as_raw(),
                uplo.into(),
                trans.into(),
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

// Add to src/rocblas/types.rs if not already present

/// Enum for diagonal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diagonal {
    /// Non-unit triangular
    NonUnit,
    /// Unit triangular
    Unit,
}

impl From<Diagonal> for ffi::rocblas_diagonal {
    fn from(diag: Diagonal) -> Self {
        match diag {
            Diagonal::NonUnit => ffi::rocblas_diagonal__rocblas_diagonal_non_unit,
            Diagonal::Unit => ffi::rocblas_diagonal__rocblas_diagonal_unit,
        }
    }
}

impl From<ffi::rocblas_diagonal> for Diagonal {
    fn from(diag: ffi::rocblas_diagonal) -> Self {
        match diag {
            ffi::rocblas_diagonal__rocblas_diagonal_non_unit => Diagonal::NonUnit,
            ffi::rocblas_diagonal__rocblas_diagonal_unit => Diagonal::Unit,
            _ => Diagonal::NonUnit, // Default to NonUnit for unknown values
        }
    }
}