// src/rocblas/level2.rs

use crate::rocblas::bindings::_rocblas_handle;
use crate::rocblas::error::{Error, Result};
use crate::rocblas::handle::Handle;
use crate::rocblas::types::{Fill, Operation};
use crate::rocblas::{ffi, rocblas_operation};
use crate::*;

use super::level3::{HemmType, HerkType, SprType, SyrBatchedType, SyrStridedBatchedType};
use super::types::Side;

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
pub unsafe fn gemv<T>(
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
    unsafe { T::rocblas_gemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy) }
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
pub unsafe fn gemv_batched<T>(
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
    unsafe {
        T::rocblas_gemv_batched(
            handle,
            trans,
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
    }
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
pub unsafe fn gemv_strided_batched<T>(
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
    unsafe {
        T::rocblas_gemv_strided_batched(
            handle,
            trans,
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
    }
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
pub unsafe fn gbmv<T>(
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
    unsafe {
        T::rocblas_gbmv(
            handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy,
        )
    }
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
pub unsafe fn gbmv_batched<T>(
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
    unsafe {
        T::rocblas_gbmv_batched(
            handle,
            trans,
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
    }
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
pub unsafe fn gbmv_strided_batched<T>(
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
    unsafe {
        T::rocblas_gbmv_strided_batched(
            handle,
            trans,
            m,
            n,
            kl,
            ku,
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
    }
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
pub unsafe fn hbmv<T>(
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
    unsafe { T::rocblas_hbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) }
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
pub unsafe fn hbmv_batched<T>(
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
    unsafe {
        T::rocblas_hbmv_batched(
            handle,
            uplo,
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
    }
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
pub unsafe fn hbmv_strided_batched<T>(
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
    unsafe {
        T::rocblas_hbmv_strided_batched(
            handle,
            uplo,
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
    }
}

//==============================================================================
// Type traits for implementation
//==============================================================================

impl_rocblas_traits!(
    GemvType,
    GemvFn,
    {
        f32 => ffi::rocblas_sgemv,
        f64 => ffi::rocblas_dgemv,
        ffi::rocblas_float_complex => ffi::rocblas_cgemv,
        ffi::rocblas_double_complex => ffi::rocblas_zgemv,
    },
    rocblas_gemv,
    (handle: &Handle, trans: Operation, m: i32, n: i32, alpha: &Self, A: *const Self, lda: i32, x: *const Self, incx: i32, beta: &Self, y: *mut Self, incy: i32),
    (*mut _rocblas_handle, rocblas_operation, i32, i32, *const T, *const T, i32, *const T, i32, *const T, *mut T, i32),
    (handle.as_raw(), trans.into(), m, n, alpha, A, lda, x, incx, beta, y, incy)
);

impl_rocblas_traits!(
    GemvBatchedType,
    GemvBatchedFn,
    {
        f32 => ffi::rocblas_sgemv_batched,
        f64 => ffi::rocblas_dgemv_batched,
        ffi::rocblas_float_complex => ffi::rocblas_cgemv_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zgemv_batched,
    },
    rocblas_gemv_batched,
    (handle: &Handle, trans: Operation, m: i32, n: i32, alpha: &Self, A: *const *const Self, lda: i32, x: *const *const Self, incx: i32, beta: &Self, y: *const *mut Self, incy: i32, batch_count: i32),
    (*mut _rocblas_handle, rocblas_operation, i32, i32, *const T, *const *const T, i32, *const *const T, i32, *const T, *const *mut T, i32, i32),
    (handle.as_raw(), trans.into(), m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
);

impl_rocblas_traits!(
    GemvStridedBatchedType,
    GemvStridedBatchedFn,
    {
        f32 => ffi::rocblas_sgemv_strided_batched,
        f64 => ffi::rocblas_dgemv_strided_batched,
        ffi::rocblas_float_complex => ffi::rocblas_cgemv_strided_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zgemv_strided_batched,
    },
    rocblas_gemv_strided_batched,
    (handle: &Handle,trans: Operation,m: i32,n: i32,alpha: &Self,A: *const Self,lda: i32,stride_A: i64,x: *const Self,incx: i32,stride_x: i64,beta: &Self,y: *mut Self,incy: i32,stride_y: i64,batch_count: i32),
    (*mut _rocblas_handle,rocblas_operation,i32,i32,*const T,*const T,i32,i64,*const T,i32,i64,*const T,*mut T,i32,i64,i32),
    (handle.as_raw(),trans.into(),m,n,alpha,A,lda,stride_A,x,incx,stride_x,beta,y,incy,stride_y,batch_count)
);

impl_rocblas_traits!(
    GbmvType,
    GbmvFn,
    {
        f32 => ffi::rocblas_sgbmv,
        f64 => ffi::rocblas_dgbmv,
        ffi::rocblas_float_complex => ffi::rocblas_cgbmv,
        ffi::rocblas_double_complex => ffi::rocblas_zgbmv,
    },
    rocblas_gbmv,
    (handle: &Handle,trans: Operation,m: i32,n: i32,kl: i32,ku: i32,alpha: &Self,A: *const Self,lda: i32,x: *const Self,incx: i32,beta: &Self,y: *mut Self,incy: i32),
    (*mut _rocblas_handle,rocblas_operation,i32,i32,i32,i32,*const T,*const T,i32,*const T,i32,*const T,*mut T,i32),
    (handle.as_raw(),trans.into(),m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy)
);

/// Trait for types that can be used with gbmv_batched
pub trait GbmvBatchedType {
    unsafe fn rocblas_gbmv_batched(
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
    unsafe fn rocblas_gbmv_batched(
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
    unsafe fn rocblas_gbmv_strided_batched(
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
    unsafe fn rocblas_hbmv(
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
    unsafe fn rocblas_hbmv(
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
    unsafe fn rocblas_hbmv(
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
    unsafe fn rocblas_hbmv_batched(
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
    unsafe fn rocblas_hbmv_strided_batched(
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
    unsafe fn rocblas_hbmv_batched(
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
    unsafe fn rocblas_hbmv_batched(
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
    unsafe fn rocblas_hbmv_strided_batched(
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
    unsafe fn rocblas_hbmv_strided_batched(
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
pub unsafe fn hemv<T>(
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
    unsafe { T::rocblas_hemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy) }
}

/// Wrapper for hemv_batched functions
pub unsafe fn hemv_batched<T>(
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
    unsafe {
        T::rocblas_hemv_batched(
            handle,
            uplo,
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
    }
}

/// Wrapper for hemv_strided_batched functions
pub unsafe fn hemv_strided_batched<T>(
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
    unsafe {
        T::rocblas_hemv_strided_batched(
            handle,
            uplo,
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
    }
}

/// Define the trait for the hemv operations
pub trait HemvType {
    unsafe fn rocblas_hemv(
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
    unsafe fn rocblas_hemv(
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
    unsafe fn rocblas_hemv(
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
    unsafe fn rocblas_hemv_batched(
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
    unsafe fn rocblas_hemv_batched(
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
    unsafe fn rocblas_hemv_batched(
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
    unsafe fn rocblas_hemv_strided_batched(
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
    unsafe fn rocblas_hemv_strided_batched(
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
    unsafe fn rocblas_hemv_strided_batched(
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

// Implementation for GER/GERU/GERC functions for level2.rs

/// Perform general rank-1 update
///
/// A := alpha * x * y^T + A
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `alpha` - Scalar alpha
/// * `x` - Vector x
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Vector y
/// * `incy` - Stride between consecutive elements of y
/// * `A` - Matrix A
/// * `lda` - Leading dimension of matrix A
pub unsafe fn ger<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    y: *const T,
    incy: i32,
    A: *mut T,
    lda: i32,
) -> Result<()>
where
    T: GerType,
{
    unsafe { T::rocblas_ger(handle, m, n, alpha, x, incx, y, incy, A, lda) }
}

/// Perform general rank-1 update for complex matrices (non-conjugated)
///
/// A := alpha * x * y^T + A
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `alpha` - Scalar alpha
/// * `x` - Vector x
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Vector y
/// * `incy` - Stride between consecutive elements of y
/// * `A` - Matrix A
/// * `lda` - Leading dimension of matrix A
pub unsafe fn geru<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    y: *const T,
    incy: i32,
    A: *mut T,
    lda: i32,
) -> Result<()>
where
    T: GeruType,
{
    unsafe { T::rocblas_geru(handle, m, n, alpha, x, incx, y, incy, A, lda) }
}

/// Perform general rank-1 update for complex matrices (conjugated)
///
/// A := alpha * x * y^H + A
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `alpha` - Scalar alpha
/// * `x` - Vector x
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Vector y
/// * `incy` - Stride between consecutive elements of y
/// * `A` - Matrix A
/// * `lda` - Leading dimension of matrix A
pub unsafe fn gerc<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    y: *const T,
    incy: i32,
    A: *mut T,
    lda: i32,
) -> Result<()>
where
    T: GercType,
{
    unsafe { T::rocblas_gerc(handle, m, n, alpha, x, incx, y, incy, A, lda) }
}

// Batched versions
pub unsafe fn ger_batched<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    alpha: &T,
    x: *const *const T,
    incx: i32,
    y: *const *const T,
    incy: i32,
    A: *const *mut T,
    lda: i32,
    batch_count: i32,
) -> Result<()>
where
    T: GerBatchedType,
{
    unsafe { T::rocblas_ger_batched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count) }
}

pub unsafe fn geru_batched<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    alpha: &T,
    x: *const *const T,
    incx: i32,
    y: *const *const T,
    incy: i32,
    A: *const *mut T,
    lda: i32,
    batch_count: i32,
) -> Result<()>
where
    T: GeruBatchedType,
{
    unsafe { T::rocblas_geru_batched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count) }
}

pub unsafe fn gerc_batched<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    alpha: &T,
    x: *const *const T,
    incx: i32,
    y: *const *const T,
    incy: i32,
    A: *const *mut T,
    lda: i32,
    batch_count: i32,
) -> Result<()>
where
    T: GercBatchedType,
{
    unsafe { T::rocblas_gerc_batched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count) }
}

// Strided batched versions
pub unsafe fn ger_strided_batched<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    stride_x: i64,
    y: *const T,
    incy: i32,
    stride_y: i64,
    A: *mut T,
    lda: i32,
    stride_A: i64,
    batch_count: i32,
) -> Result<()>
where
    T: GerStridedBatchedType,
{
    unsafe {
        T::rocblas_ger_strided_batched(
            handle,
            m,
            n,
            alpha,
            x,
            incx,
            stride_x,
            y,
            incy,
            stride_y,
            A,
            lda,
            stride_A,
            batch_count,
        )
    }
}

pub unsafe fn geru_strided_batched<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    stride_x: i64,
    y: *const T,
    incy: i32,
    stride_y: i64,
    A: *mut T,
    lda: i32,
    stride_A: i64,
    batch_count: i32,
) -> Result<()>
where
    T: GeruStridedBatchedType,
{
    unsafe {
        T::rocblas_geru_strided_batched(
            handle,
            m,
            n,
            alpha,
            x,
            incx,
            stride_x,
            y,
            incy,
            stride_y,
            A,
            lda,
            stride_A,
            batch_count,
        )
    }
}

pub unsafe fn gerc_strided_batched<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    stride_x: i64,
    y: *const T,
    incy: i32,
    stride_y: i64,
    A: *mut T,
    lda: i32,
    stride_A: i64,
    batch_count: i32,
) -> Result<()>
where
    T: GercStridedBatchedType,
{
    unsafe {
        T::rocblas_gerc_strided_batched(
            handle,
            m,
            n,
            alpha,
            x,
            incx,
            stride_x,
            y,
            incy,
            stride_y,
            A,
            lda,
            stride_A,
            batch_count,
        )
    }
}

// Trait definitions for GER operations
pub trait GerType {
    unsafe fn rocblas_ger(
        handle: &Handle,
        m: i32,
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

impl GerType for f32 {
    unsafe fn rocblas_ger(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_sger(handle.as_raw(), m, n, alpha, x, incx, y, incy, A, lda) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GerType for f64 {
    unsafe fn rocblas_ger(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_dger(handle.as_raw(), m, n, alpha, x, incx, y, incy, A, lda) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

pub trait GeruType {
    unsafe fn rocblas_geru(
        handle: &Handle,
        m: i32,
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

impl GeruType for ffi::rocblas_float_complex {
    unsafe fn rocblas_geru(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_cgeru(handle.as_raw(), m, n, alpha, x, incx, y, incy, A, lda) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GeruType for ffi::rocblas_double_complex {
    unsafe fn rocblas_geru(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_zgeru(handle.as_raw(), m, n, alpha, x, incx, y, incy, A, lda) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

pub trait GercType {
    unsafe fn rocblas_gerc(
        handle: &Handle,
        m: i32,
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

impl GercType for ffi::rocblas_float_complex {
    unsafe fn rocblas_gerc(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_cgerc(handle.as_raw(), m, n, alpha, x, incx, y, incy, A, lda) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GercType for ffi::rocblas_double_complex {
    unsafe fn rocblas_gerc(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_zgerc(handle.as_raw(), m, n, alpha, x, incx, y, incy, A, lda) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Batched trait implementations
pub trait GerBatchedType {
    unsafe fn rocblas_ger_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()>;
}

// Similar implementations for GerBatchedType, GeruBatchedType, GercBatchedType,
// GerStridedBatchedType, GeruStridedBatchedType, GercStridedBatchedType

// Implementations for SPR/SPR2 functions (symmetric rank-1/rank-2 updates with packed storage)
/// Perform symmetric rank-1 update with packed storage
///
/// A := alpha * x * x^T + A
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of matrix A
/// * `alpha` - Scalar alpha
/// * `x` - Vector x
/// * `incx` - Stride between consecutive elements of x
/// * `AP` - Packed matrix A
pub unsafe fn spr<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    AP: *mut T,
) -> Result<()>
where
    T: SprType,
{
    T::rocblas_spr(handle, uplo, n, alpha, x, incx, AP)
}

// Similar functions and traits for spr, spr2, syr, syr2

// For level3.rs additions
/// Hermitian matrix-matrix multiplication
///
/// C := alpha * A * B + beta * C  if side == Side::Left
/// C := alpha * B * A + beta * C  if side == Side::Right
///
/// where alpha and beta are scalars, A is a Hermitian matrix, and B and C are m by n matrices.
pub unsafe fn hemm<T>(
    handle: &Handle,
    side: Side,
    uplo: Fill,
    m: i32,
    n: i32,
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
    T: HemmType,
{
    T::rocblas_hemm(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc,
    )
}

/// Hermitian rank-k update
///
/// C := alpha * A * A^H + beta * C  if transA == Operation::None
/// C := alpha * A^H * A + beta * C  if transA == Operation::ConjugateTranspose
///
/// where alpha and beta are scalars, C is an n by n Hermitian matrix and A is an n by k matrix in the
/// first case and a k by n
///
pub unsafe fn herk<T, R>(
    handle: &Handle,
    uplo: Fill,
    transA: Operation,
    n: i32,
    k: i32,
    alpha: &R,
    A: *const T,
    lda: i32,
    beta: &R,
    C: *mut T,
    ldc: i32,
) -> Result<()>
where
    T: HerkType<ScalarType = R>,
{
    T::rocblas_herk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc)
}

pub unsafe fn syr_batched<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    alpha: &T,
    x: *const *const T,
    incx: i32,
    A: *const *mut T,
    lda: i32,
    batch_count: i32,
) -> Result<()>
where
    T: SyrBatchedType,
{
    T::rocblas_syr_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
}

/// Strided batched symmetric rank-1 update
pub unsafe fn syr_strided_batched<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    stride_x: i64,
    A: *mut T,
    lda: i32,
    stride_A: i64,
    batch_count: i32,
) -> Result<()>
where
    T: SyrStridedBatchedType,
{
    T::rocblas_syr_strided_batched(
        handle,
        uplo,
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
}

/// Batched symmetric rank-2 update
pub unsafe fn syr2_batched<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    alpha: &T,
    x: *const *const T,
    incx: i32,
    y: *const *const T,
    incy: i32,
    A: *const *mut T,
    lda: i32,
    batch_count: i32,
) -> Result<()>
where
    T: Syr2BatchedType,
{
    unsafe {
        T::rocblas_syr2_batched(
            handle,
            uplo,
            n,
            alpha,
            x,
            incx,
            y,
            incy,
            A,
            lda,
            batch_count,
        )
    }
}

/// Strided batched symmetric rank-2 update
pub unsafe fn syr2_strided_batched<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    stride_x: i64,
    y: *const T,
    incy: i32,
    stride_y: i64,
    A: *mut T,
    lda: i32,
    stride_A: i64,
    batch_count: i32,
) -> Result<()>
where
    T: Syr2StridedBatchedType,
{
    unsafe {
        T::rocblas_syr2_strided_batched(
            handle,
            uplo,
            n,
            alpha,
            x,
            incx,
            stride_x,
            y,
            incy,
            stride_y,
            A,
            lda,
            stride_A,
            batch_count,
        )
    }
}

/// Batched packed symmetric rank-1 update
pub unsafe fn spr_batched<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    alpha: &T,
    x: *const *const T,
    incx: i32,
    AP: *const *mut T,
    batch_count: i32,
) -> Result<()>
where
    T: SprBatchedType,
{
    unsafe { T::rocblas_spr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count) }
}

/// Strided batched packed symmetric rank-1 update
pub unsafe fn spr_strided_batched<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    stride_x: i64,
    AP: *mut T,
    stride_A: i64,
    batch_count: i32,
) -> Result<()>
where
    T: SprStridedBatchedType,
{
    unsafe {
        T::rocblas_spr_strided_batched(
            handle,
            uplo,
            n,
            alpha,
            x,
            incx,
            stride_x,
            AP,
            stride_A,
            batch_count,
        )
    }
}

/// Batched packed symmetric rank-2 update
pub unsafe fn spr2_batched<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    alpha: &T,
    x: *const *const T,
    incx: i32,
    y: *const *const T,
    incy: i32,
    AP: *const *mut T,
    batch_count: i32,
) -> Result<()>
where
    T: Spr2BatchedType,
{
    unsafe { T::rocblas_spr2_batched(handle, uplo, n, alpha, x, incx, y, incy, AP, batch_count) }
}

/// Strided batched packed symmetric rank-2 update
pub unsafe fn spr2_strided_batched<T>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    stride_x: i64,
    y: *const T,
    incy: i32,
    stride_y: i64,
    AP: *mut T,
    stride_A: i64,
    batch_count: i32,
) -> Result<()>
where
    T: Spr2StridedBatchedType,
{
    unsafe {
        T::rocblas_spr2_strided_batched(
            handle,
            uplo,
            n,
            alpha,
            x,
            incx,
            stride_x,
            y,
            incy,
            stride_y,
            AP,
            stride_A,
            batch_count,
        )
    }
}

// Missing trait definitions for GER batched and strided batched operations

impl GerBatchedType for f32 {
    unsafe fn rocblas_ger_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sger_batched(
                handle.as_raw(),
                m,
                n,
                alpha,
                x,
                incx,
                y,
                incy,
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

impl GerBatchedType for f64 {
    unsafe fn rocblas_ger_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dger_batched(
                handle.as_raw(),
                m,
                n,
                alpha,
                x,
                incx,
                y,
                incy,
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

pub trait GeruBatchedType {
    unsafe fn rocblas_geru_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl GeruBatchedType for ffi::rocblas_float_complex {
    unsafe fn rocblas_geru_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgeru_batched(
                handle.as_raw(),
                m,
                n,
                alpha,
                x,
                incx,
                y,
                incy,
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

impl GeruBatchedType for ffi::rocblas_double_complex {
    unsafe fn rocblas_geru_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgeru_batched(
                handle.as_raw(),
                m,
                n,
                alpha,
                x,
                incx,
                y,
                incy,
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

pub trait GercBatchedType {
    unsafe fn rocblas_gerc_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl GercBatchedType for ffi::rocblas_float_complex {
    unsafe fn rocblas_gerc_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgerc_batched(
                handle.as_raw(),
                m,
                n,
                alpha,
                x,
                incx,
                y,
                incy,
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

impl GercBatchedType for ffi::rocblas_double_complex {
    unsafe fn rocblas_gerc_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgerc_batched(
                handle.as_raw(),
                m,
                n,
                alpha,
                x,
                incx,
                y,
                incy,
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

pub trait GerStridedBatchedType {
    unsafe fn rocblas_ger_strided_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl GerStridedBatchedType for f32 {
    unsafe fn rocblas_ger_strided_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sger_strided_batched(
                handle.as_raw(),
                m,
                n,
                alpha,
                x,
                incx,
                stride_x,
                y,
                incy,
                stride_y,
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

impl GerStridedBatchedType for f64 {
    unsafe fn rocblas_ger_strided_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dger_strided_batched(
                handle.as_raw(),
                m,
                n,
                alpha,
                x,
                incx,
                stride_x,
                y,
                incy,
                stride_y,
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

pub trait GeruStridedBatchedType {
    unsafe fn rocblas_geru_strided_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl GeruStridedBatchedType for ffi::rocblas_float_complex {
    unsafe fn rocblas_geru_strided_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgeru_strided_batched(
                handle.as_raw(),
                m,
                n,
                alpha,
                x,
                incx,
                stride_x,
                y,
                incy,
                stride_y,
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

impl GeruStridedBatchedType for ffi::rocblas_double_complex {
    unsafe fn rocblas_geru_strided_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgeru_strided_batched(
                handle.as_raw(),
                m,
                n,
                alpha,
                x,
                incx,
                stride_x,
                y,
                incy,
                stride_y,
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

pub trait GercStridedBatchedType {
    unsafe fn rocblas_gerc_strided_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl GercStridedBatchedType for ffi::rocblas_float_complex {
    unsafe fn rocblas_gerc_strided_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgerc_strided_batched(
                handle.as_raw(),
                m,
                n,
                alpha,
                x,
                incx,
                stride_x,
                y,
                incy,
                stride_y,
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

impl GercStridedBatchedType for ffi::rocblas_double_complex {
    unsafe fn rocblas_gerc_strided_batched(
        handle: &Handle,
        m: i32,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgerc_strided_batched(
                handle.as_raw(),
                m,
                n,
                alpha,
                x,
                incx,
                stride_x,
                y,
                incy,
                stride_y,
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

// Traits for SPR batched and strided batched operations
pub trait SprBatchedType {
    unsafe fn rocblas_spr_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        AP: *const *mut Self,
        batch_count: i32,
    ) -> Result<()>;
}

impl SprBatchedType for f32 {
    unsafe fn rocblas_spr_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        AP: *const *mut Self,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sspr_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                AP,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl SprBatchedType for f64 {
    unsafe fn rocblas_spr_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        AP: *const *mut Self,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dspr_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                AP,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl SprBatchedType for ffi::rocblas_float_complex {
    unsafe fn rocblas_spr_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        AP: *const *mut Self,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cspr_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                AP,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl SprBatchedType for ffi::rocblas_double_complex {
    unsafe fn rocblas_spr_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        AP: *const *mut Self,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zspr_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                AP,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

pub trait SprStridedBatchedType {
    unsafe fn rocblas_spr_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        AP: *mut Self,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl SprStridedBatchedType for f32 {
    unsafe fn rocblas_spr_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        AP: *mut Self,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sspr_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                stride_x,
                AP,
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

impl SprStridedBatchedType for f64 {
    unsafe fn rocblas_spr_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        AP: *mut Self,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dspr_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                stride_x,
                AP,
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

impl SprStridedBatchedType for ffi::rocblas_float_complex {
    unsafe fn rocblas_spr_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        AP: *mut Self,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cspr_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                stride_x,
                AP,
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

impl SprStridedBatchedType for ffi::rocblas_double_complex {
    unsafe fn rocblas_spr_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        AP: *mut Self,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zspr_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                stride_x,
                AP,
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

// Traits for SPR2 batched and strided batched operations
pub trait Spr2BatchedType {
    unsafe fn rocblas_spr2_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        AP: *const *mut Self,
        batch_count: i32,
    ) -> Result<()>;
}

impl Spr2BatchedType for f32 {
    unsafe fn rocblas_spr2_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        AP: *const *mut Self,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sspr2_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                AP,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl Spr2BatchedType for f64 {
    unsafe fn rocblas_spr2_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        AP: *const *mut Self,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dspr2_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                AP,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

pub trait Spr2StridedBatchedType {
    unsafe fn rocblas_spr2_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        AP: *mut Self,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl Spr2StridedBatchedType for f32 {
    unsafe fn rocblas_spr2_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        AP: *mut Self,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sspr2_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                stride_x,
                y,
                incy,
                stride_y,
                AP,
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

impl Spr2StridedBatchedType for f64 {
    unsafe fn rocblas_spr2_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        AP: *mut Self,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dspr2_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                stride_x,
                y,
                incy,
                stride_y,
                AP,
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

// Traits for SYR2 batched and strided batched operations
pub trait Syr2BatchedType {
    unsafe fn rocblas_syr2_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl Syr2BatchedType for f32 {
    unsafe fn rocblas_syr2_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ssyr2_batched(
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
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl Syr2BatchedType for f64 {
    unsafe fn rocblas_syr2_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dsyr2_batched(
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
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl Syr2BatchedType for ffi::rocblas_float_complex {
    unsafe fn rocblas_syr2_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_csyr2_batched(
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
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl Syr2BatchedType for ffi::rocblas_double_complex {
    unsafe fn rocblas_syr2_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        y: *const *const Self,
        incy: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zsyr2_batched(
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
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

pub trait Syr2StridedBatchedType {
    unsafe fn rocblas_syr2_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl Syr2StridedBatchedType for f32 {
    unsafe fn rocblas_syr2_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ssyr2_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                stride_x,
                y,
                incy,
                stride_y,
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

impl Syr2StridedBatchedType for f64 {
    unsafe fn rocblas_syr2_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dsyr2_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                stride_x,
                y,
                incy,
                stride_y,
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

impl Syr2StridedBatchedType for ffi::rocblas_float_complex {
    unsafe fn rocblas_syr2_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_csyr2_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                stride_x,
                y,
                incy,
                stride_y,
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

impl Syr2StridedBatchedType for ffi::rocblas_double_complex {
    unsafe fn rocblas_syr2_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        y: *const Self,
        incy: i32,
        stride_y: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zsyr2_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                stride_x,
                y,
                incy,
                stride_y,
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
