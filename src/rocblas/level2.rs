// src/rocblas/level2.rs

use crate::rocblas::ffi;
use crate::rocblas::handle::Handle;
use crate::rocblas::error::{Error, Result};
use crate::rocblas::types::{Operation, Fill};

//==============================================================================
// GEMV functions - General Matrix-Vector Multiplication
//==============================================================================

/// Matrix-vector multiplication with a general matrix
/// 
/// Computes one of the following matrix-vector operations:
/// 
/// y := alpha * A * x + beta * y
/// y := alpha * A^T * x + beta * y
/// y := alpha * A^H * x + beta * y
/// 
/// where alpha and beta are scalars, x and y are vectors, and A is an m x n matrix.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `trans` - Operation op(A) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `alpha` - Scalar alpha
/// * `A` - Buffer storing matrix A
/// * `lda` - Leading dimension of matrix A
/// * `x` - Buffer storing vector x
/// * `incx` - Stride between consecutive elements of x
/// * `beta` - Scalar beta
/// * `y` - Buffer storing vector y
/// * `incy` - Stride between consecutive elements of y
pub fn gemv<T>(
    handle: &Handle,
    trans: Operation,
    m: i32,
    n: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    x: *const T,
    incx: i32,
    beta: &T,
    y: *mut T,
    incy: i32,
) -> Result<()>
where
    T: GemvType,
{
    T::rocblas_gemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
}

/// Batched matrix-vector multiplication with general matrices
/// 
/// Computes one of the following batched matrix-vector operations:
/// 
/// y_i := alpha * A_i * x_i + beta * y_i
/// y_i := alpha * A_i^T * x_i + beta * y_i
/// y_i := alpha * A_i^H * x_i + beta * y_i
/// 
/// where (A_i, x_i, y_i) is the i-th instance of the batch.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `trans` - Operation op(A) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrices A_i
/// * `n` - Number of columns of matrices A_i
/// * `alpha` - Scalar alpha
/// * `A` - Array of pointers to matrices A_i
/// * `lda` - Leading dimension of matrices A_i
/// * `x` - Array of pointers to vectors x_i
/// * `incx` - Stride between consecutive elements of vectors x_i
/// * `beta` - Scalar beta
/// * `y` - Array of pointers to vectors y_i
/// * `incy` - Stride between consecutive elements of vectors y_i
/// * `batch_count` - Number of instances in the batch
pub fn gemv_batched<T>(
    handle: &Handle,
    trans: Operation,
    m: i32,
    n: i32,
    alpha: &T,
    A: *const *const T,
    lda: i32,
    x: *const *const T,
    incx: i32,
    beta: &T,
    y: *const *mut T,
    incy: i32,
    batch_count: i32,
) -> Result<()>
where
    T: GemvBatchedType,
{
    T::rocblas_gemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
}

/// Strided batched matrix-vector multiplication with general matrices
/// 
/// Computes one of the following strided batched matrix-vector operations:
/// 
/// y_i := alpha * A_i * x_i + beta * y_i
/// y_i := alpha * A_i^T * x_i + beta * y_i
/// y_i := alpha * A_i^H * x_i + beta * y_i
/// 
/// where (A_i, x_i, y_i) is the i-th instance of the batch.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `trans` - Operation op(A) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrices A_i
/// * `n` - Number of columns of matrices A_i
/// * `alpha` - Scalar alpha
/// * `A` - Pointer to the first matrix A_1
/// * `lda` - Leading dimension of matrices A_i
/// * `stride_A` - Stride from start of one matrix (A_i) to the next (A_i+1)
/// * `x` - Pointer to the first vector x_1
/// * `incx` - Stride between consecutive elements of vectors x_i
/// * `stride_x` - Stride from start of one vector (x_i) to the next (x_i+1)
/// * `beta` - Scalar beta
/// * `y` - Pointer to the first vector y_1
/// * `incy` - Stride between consecutive elements of vectors y_i
/// * `stride_y` - Stride from start of one vector (y_i) to the next (y_i+1)
/// * `batch_count` - Number of instances in the batch
pub fn gemv_strided_batched<T>(
    handle: &Handle,
    trans: Operation,
    m: i32,
    n: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    stride_A: i64,
    x: *const T,
    incx: i32,
    stride_x: i64,
    beta: &T,
    y: *mut T,
    incy: i32,
    stride_y: i64,
    batch_count: i32,
) -> Result<()>
where
    T: GemvStridedBatchedType,
{
    T::rocblas_gemv_strided_batched(
        handle, trans, m, n, alpha, A, lda, stride_A, 
        x, incx, stride_x, beta, y, incy, stride_y, batch_count
    )
}

//==============================================================================
// GBMV functions - General Banded Matrix-Vector Multiplication
//==============================================================================

/// Matrix-vector multiplication with a general banded matrix
/// 
/// Computes one of the following matrix-vector operations:
/// 
/// y := alpha * A * x + beta * y
/// y := alpha * A^T * x + beta * y
/// y := alpha * A^H * x + beta * y
/// 
/// where alpha and beta are scalars, x and y are vectors, and A is an m x n banded matrix
/// with kl sub-diagonals and ku super-diagonals.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `trans` - Operation op(A) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `kl` - Number of sub-diagonals of matrix A
/// * `ku` - Number of super-diagonals of matrix A
/// * `alpha` - Scalar alpha
/// * `A` - Buffer storing matrix A
/// * `lda` - Leading dimension of matrix A
/// * `x` - Buffer storing vector x
/// * `incx` - Stride between consecutive elements of x
/// * `beta` - Scalar beta
/// * `y` - Buffer storing vector y
/// * `incy` - Stride between consecutive elements of y
pub fn gbmv<T>(
    handle: &Handle,
    trans: Operation,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    x: *const T,
    incx: i32,
    beta: &T,
    y: *mut T,
    incy: i32,
) -> Result<()>
where
    T: GbmvType,
{
    T::rocblas_gbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
}

/// Batched matrix-vector multiplication with general banded matrices
/// 
/// Computes one of the following batched matrix-vector operations:
/// 
/// y_i := alpha * A_i * x_i + beta * y_i
/// y_i := alpha * A_i^T * x_i + beta * y_i
/// y_i := alpha * A_i^H * x_i + beta * y_i
/// 
/// where (A_i, x_i, y_i) is the i-th instance of the batch. A_i is a banded matrix
/// with kl sub-diagonals and ku super-diagonals.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `trans` - Operation op(A) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrices A_i
/// * `n` - Number of columns of matrices A_i
/// * `kl` - Number of sub-diagonals of matrices A_i
/// * `ku` - Number of super-diagonals of matrices A_i
/// * `alpha` - Scalar alpha
/// * `A` - Array of pointers to matrices A_i
/// * `lda` - Leading dimension of matrices A_i
/// * `x` - Array of pointers to vectors x_i
/// * `incx` - Stride between consecutive elements of vectors x_i
/// * `beta` - Scalar beta
/// * `y` - Array of pointers to vectors y_i
/// * `incy` - Stride between consecutive elements of vectors y_i
/// * `batch_count` - Number of instances in the batch
pub fn gbmv_batched<T>(
    handle: &Handle,
    trans: Operation,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: &T,
    A: *const *const T,
    lda: i32,
    x: *const *const T,
    incx: i32,
    beta: &T,
    y: *const *mut T,
    incy: i32,
    batch_count: i32,
) -> Result<()>
where
    T: GbmvBatchedType,
{
    T::rocblas_gbmv_batched(
        handle, trans, m, n, kl, ku, alpha, A, lda, 
        x, incx, beta, y, incy, batch_count
    )
}

/// Strided batched matrix-vector multiplication with general banded matrices
/// 
/// Computes one of the following strided batched matrix-vector operations:
/// 
/// y_i := alpha * A_i * x_i + beta * y_i
/// y_i := alpha * A_i^T * x_i + beta * y_i
/// y_i := alpha * A_i^H * x_i + beta * y_i
/// 
/// where (A_i, x_i, y_i) is the i-th instance of the batch. A_i is a banded matrix
/// with kl sub-diagonals and ku super-diagonals.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `trans` - Operation op(A) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrices A_i
/// * `n` - Number of columns of matrices A_i
/// * `kl` - Number of sub-diagonals of matrices A_i
/// * `ku` - Number of super-diagonals of matrices A_i
/// * `alpha` - Scalar alpha
/// * `A` - Pointer to the first matrix A_1
/// * `lda` - Leading dimension of matrices A_i
/// * `stride_A` - Stride from start of one matrix (A_i) to the next (A_i+1)
/// * `x` - Pointer to the first vector x_1
/// * `incx` - Stride between consecutive elements of vectors x_i
/// * `stride_x` - Stride from start of one vector (x_i) to the next (x_i+1)
/// * `beta` - Scalar beta
/// * `y` - Pointer to the first vector y_1
/// * `incy` - Stride between consecutive elements of vectors y_i
/// * `stride_y` - Stride from start of one vector (y_i) to the next (y_i+1)
/// * `batch_count` - Number of instances in the batch
pub fn gbmv_strided_batched<T>(
    handle: &Handle,
    trans: Operation,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    stride_A: i64,
    x: *const T,
    incx: i32,
    stride_x: i64,
    beta: &T,
    y: *mut T,
    incy: i32,
    stride_y: i64,
    batch_count: i32,
) -> Result<()>
where
    T: GbmvStridedBatchedType,
{
    T::rocblas_gbmv_strided_batched(
        handle, trans, m, n, kl, ku, alpha, A, lda, stride_A,
        x, incx, stride_x, beta, y, incy, stride_y, batch_count
    )
}

//==============================================================================
// HBMV functions - Hermitian Banded Matrix-Vector Multiplication
//==============================================================================

/// Matrix-vector multiplication with a Hermitian banded matrix
/// 
/// Computes the following matrix-vector operation:
/// 
/// y := alpha * A * x + beta * y
/// 
/// where alpha and beta are scalars, x and y are vectors, and A is an n x n Hermitian banded matrix
/// with k super-diagonals.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part of A is used
/// * `n` - Number of rows and columns of matrix A
/// * `k` - Number of super-diagonals of matrix A
/// * `alpha` - Scalar alpha
/// * `A` - Buffer storing matrix A
/// * `lda` - Leading dimension of matrix A
/// * `x` - Buffer storing vector x
/// * `incx` - Stride between consecutive elements of x
/// * `beta` - Scalar beta
/// * `y` - Buffer storing vector y
/// * `incy` - Stride between consecutive elements of y
pub fn hbmv<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    x: *const T,
    incx: i32,
    beta: &T,
    y: *mut T,
    incy: i32,
) -> Result<()>
where
    T: HbmvType,
{
    T::rocblas_hbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
}

/// Batched matrix-vector multiplication with Hermitian banded matrices
/// 
/// Computes the following batched matrix-vector operation:
/// 
/// y_i := alpha * A_i * x_i + beta * y_i
/// 
/// where (A_i, x_i, y_i) is the i-th instance of the batch. A_i is a Hermitian banded matrix.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part of A_i is used
/// * `n` - Number of rows and columns of matrices A_i
/// * `k` - Number of super-diagonals of matrices A_i
/// * `alpha` - Scalar alpha
/// * `A` - Array of pointers to matrices A_i
/// * `lda` - Leading dimension of matrices A_i
/// * `x` - Array of pointers to vectors x_i
/// * `incx` - Stride between consecutive elements of vectors x_i
/// * `beta` - Scalar beta
/// * `y` - Array of pointers to vectors y_i
/// * `incy` - Stride between consecutive elements of vectors y_i
/// * `batch_count` - Number of instances in the batch
pub fn hbmv_batched<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const *const T,
    lda: i32,
    x: *const *const T,
    incx: i32,
    beta: &T,
    y: *const *mut T,
    incy: i32,
    batch_count: i32,
) -> Result<()>
where
    T: HbmvBatchedType,
{
    T::rocblas_hbmv_batched(
        handle, uplo, n, k, alpha, A, lda, 
        x, incx, beta, y, incy, batch_count
    )
}

/// Strided batched matrix-vector multiplication with Hermitian banded matrices
/// 
/// Computes the following strided batched matrix-vector operation:
/// 
/// y_i := alpha * A_i * x_i + beta * y_i
/// 
/// where (A_i, x_i, y_i) is the i-th instance of the batch. A_i is a Hermitian banded matrix.
/// 
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part of A_i is used
/// * `n` - Number of rows and columns of matrices A_i
/// * `k` - Number of super-diagonals of matrices A_i
/// * `alpha` - Scalar alpha
/// * `A` - Pointer to the first matrix A_1
/// * `lda` - Leading dimension of matrices A_i
/// * `stride_A` - Stride from start of one matrix (A_i) to the next (A_i+1)
/// * `x` - Pointer to the first vector x_1
/// * `incx` - Stride between consecutive elements of vectors x_i
/// * `stride_x` - Stride from start of one vector (x_i) to the next (x_i+1)
/// * `beta` - Scalar beta
/// * `y` - Pointer to the first vector y_1
/// * `incy` - Stride between consecutive elements of vectors y_i
/// * `stride_y` - Stride from start of one vector (y_i) to the next (y_i+1)
/// * `batch_count` - Number of instances in the batch
pub fn hbmv_strided_batched<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    stride_A: i64,
    x: *const T,
    incx: i32,
    stride_x: i64,
    beta: &T,
    y: *mut T,
    incy: i32,
    stride_y: i64,
    batch_count: i32,
) -> Result<()>
where
    T: HbmvStridedBatchedType,
{
    T::rocblas_hbmv_strided_batched(
        handle, uplo, n, k, alpha, A, lda, stride_A,
        x, incx, stride_x, beta, y, incy, stride_y, batch_count
    )
}

//==============================================================================
// Type traits for implementation
//==============================================================================

/// Trait for types that can be used with gemv
pub trait GemvType {
    fn rocblas_gemv(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()>;
}

impl GemvType for f32 {
    fn rocblas_gemv(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sgemv(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemvType for f64 {
    fn rocblas_gemv(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dgemv(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemvType for ffi::rocblas_float_complex {
    fn rocblas_gemv(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgemv(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemvType for ffi::rocblas_double_complex {
    fn rocblas_gemv(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgemv(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with gemv_batched
pub trait GemvBatchedType {
    fn rocblas_gemv_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl GemvBatchedType for f32 {
    fn rocblas_gemv_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sgemv_batched(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemvBatchedType for f64 {
    fn rocblas_gemv_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dgemv_batched(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemvBatchedType for ffi::rocblas_float_complex {
    fn rocblas_gemv_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgemv_batched(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemvBatchedType for ffi::rocblas_double_complex {
    fn rocblas_gemv_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgemv_batched(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with gemv_strided_batched
pub trait GemvStridedBatchedType {
    fn rocblas_gemv_strided_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        beta: &Self,
        y: *mut Self,
        incy: i32,
        stride_y: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl GemvStridedBatchedType for f32 {
    fn rocblas_gemv_strided_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        beta: &Self,
        y: *mut Self,
        incy: i32,
        stride_y: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sgemv_strided_batched(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                alpha,
                A,
                lda,
                stride_A,
                x,
                incx,
                stride_x,
                beta,
                y,
                incy,
                stride_y,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemvStridedBatchedType for f64 {
    fn rocblas_gemv_strided_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        beta: &Self,
        y: *mut Self,
        incy: i32,
        stride_y: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dgemv_strided_batched(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                alpha,
                A,
                lda,
                stride_A,
                x,
                incx,
                stride_x,
                beta,
                y,
                incy,
                stride_y,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemvStridedBatchedType for ffi::rocblas_float_complex {
    fn rocblas_gemv_strided_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        beta: &Self,
        y: *mut Self,
        incy: i32,
        stride_y: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgemv_strided_batched(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                alpha,
                A,
                lda,
                stride_A,
                x,
                incx,
                stride_x,
                beta,
                y,
                incy,
                stride_y,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemvStridedBatchedType for ffi::rocblas_double_complex {
    fn rocblas_gemv_strided_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        beta: &Self,
        y: *mut Self,
        incy: i32,
        stride_y: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgemv_strided_batched(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                alpha,
                A,
                lda,
                stride_A,
                x,
                incx,
                stride_x,
                beta,
                y,
                incy,
                stride_y,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with gbmv
pub trait GbmvType {
    fn rocblas_gbmv(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()>;
}

impl GbmvType for f32 {
    fn rocblas_gbmv(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sgbmv(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GbmvType for f64 {
    fn rocblas_gbmv(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dgbmv(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GbmvType for ffi::rocblas_float_complex {
    fn rocblas_gbmv(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgbmv(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GbmvType for ffi::rocblas_double_complex {
    fn rocblas_gbmv(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgbmv(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with gbmv_batched
pub trait GbmvBatchedType {
    fn rocblas_gbmv_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl GbmvBatchedType for f32 {
    fn rocblas_gbmv_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sgbmv_batched(
                handle.as_raw(),
                trans.into(),
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with gbmv_strided_batched
pub trait GbmvStridedBatchedType {
    fn rocblas_gbmv_strided_batched(
        handle: &Handle,
        trans: Operation,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        beta: &Self,
        y: *mut Self,
        incy: i32,
        stride_y: i64,
        batch_count: i32,
    ) -> Result<()>;
}

/// Trait for types that can be used with hbmv
pub trait HbmvType {
    fn rocblas_hbmv(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()>;
}

impl HbmvType for ffi::rocblas_float_complex {
    fn rocblas_hbmv(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_chbmv(
                handle.as_raw(),
                uplo.into(),
                n,
                k,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HbmvType for ffi::rocblas_double_complex {
    fn rocblas_hbmv(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zhbmv(
                handle.as_raw(),
                uplo.into(),
                n,
                k,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with hbmv_batched
pub trait HbmvBatchedType {
    fn rocblas_hbmv_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()>;
}

/// Trait for types that can be used with hbmv_strided_batched
pub trait HbmvStridedBatchedType {
    fn rocblas_hbmv_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        beta: &Self,
        y: *mut Self,
        incy: i32,
        stride_y: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl HbmvBatchedType for ffi::rocblas_float_complex {
    fn rocblas_hbmv_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_chbmv_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                k,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HbmvBatchedType for ffi::rocblas_double_complex {
    fn rocblas_hbmv_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zhbmv_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                k,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HbmvStridedBatchedType for ffi::rocblas_float_complex {
    fn rocblas_hbmv_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        beta: &Self,
        y: *mut Self,
        incy: i32,
        stride_y: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_chbmv_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                x,
                incx,
                stride_x,
                beta,
                y,
                incy,
                stride_y,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HbmvStridedBatchedType for ffi::rocblas_double_complex {
    fn rocblas_hbmv_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        beta: &Self,
        y: *mut Self,
        incy: i32,
        stride_y: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zhbmv_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                x,
                incx,
                stride_x,
                beta,
                y,
                incy,
                stride_y,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Implement the high-level wrapper functions for the Hermitian matrix operations

/// Wrapper for hemv functions
pub fn hemv<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    x: *const T,
    incx: i32,
    beta: &T,
    y: *mut T,
    incy: i32,
) -> Result<()>
where
    T: HemvType,
{
    T::rocblas_hemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
}

/// Wrapper for hemv_batched functions
pub fn hemv_batched<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    alpha: &T,
    A: *const *const T,
    lda: i32,
    x: *const *const T,
    incx: i32,
    beta: &T,
    y: *const *mut T,
    incy: i32,
    batch_count: i32,
) -> Result<()>
where
    T: HemvBatchedType,
{
    T::rocblas_hemv_batched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
}

/// Wrapper for hemv_strided_batched functions
pub fn hemv_strided_batched<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    stride_A: i64,
    x: *const T,
    incx: i32,
    stride_x: i64,
    beta: &T,
    y: *mut T,
    incy: i32,
    stride_y: i64,
    batch_count: i32,
) -> Result<()>
where
    T: HemvStridedBatchedType,
{
    T::rocblas_hemv_strided_batched(
        handle, uplo, n, alpha, A, lda, stride_A, 
        x, incx, stride_x, beta, y, incy, stride_y, batch_count,
    )
}

/// Define the trait for the hemv operations
pub trait HemvType {
    fn rocblas_hemv(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()>;
}

impl HemvType for ffi::rocblas_float_complex {
    fn rocblas_hemv(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_chemv(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HemvType for ffi::rocblas_double_complex {
    fn rocblas_hemv(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: &Self,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zhemv(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Define the trait for the hemv_batched operations
pub trait HemvBatchedType {
    fn rocblas_hemv_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl HemvBatchedType for ffi::rocblas_float_complex {
    fn rocblas_hemv_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_chemv_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HemvBatchedType for ffi::rocblas_double_complex {
    fn rocblas_hemv_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        x: *const *const Self,
        incx: i32,
        beta: &Self,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zhemv_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Define the trait for the hemv_strided_batched operations
pub trait HemvStridedBatchedType {
    fn rocblas_hemv_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        beta: &Self,
        y: *mut Self,
        incy: i32,
        stride_y: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl HemvStridedBatchedType for ffi::rocblas_float_complex {
    fn rocblas_hemv_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        beta: &Self,
        y: *mut Self,
        incy: i32,
        stride_y: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_chemv_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                A,
                lda,
                stride_A,
                x,
                incx,
                stride_x,
                beta,
                y,
                incy,
                stride_y,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HemvStridedBatchedType for ffi::rocblas_double_complex {
    fn rocblas_hemv_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        beta: &Self,
        y: *mut Self,
        incy: i32,
        stride_y: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zhemv_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                A,
                lda,
                stride_A,
                x,
                incx,
                stride_x,
                beta,
                y,
                incy,
                stride_y,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}