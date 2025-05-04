// src/rocblas/level1.rs

use crate::rocblas::ffi;
use crate::rocblas::handle::Handle;
use crate::rocblas::error::{Error, Result};

//==============================================================================
// SCAL functions
//==============================================================================

/// Scale a vector by a scalar
/// 
/// x := alpha * x
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in vector x
/// * `alpha` - Scalar
/// * `x` - Device pointer to vector x
/// * `incx` - Stride between consecutive elements of x
pub fn scal<T>(
    handle: &Handle,
    n: i32,
    alpha: &T,
    x: *mut T,
    incx: i32,
) -> Result<()>
where
    T: ScalType,
{
    T::rocblas_scal(handle, n, alpha, x, incx)
}

/// Scale vectors in a batch by a scalar
/// 
/// x_i := alpha * x_i, for i = 1,...,batch_count
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in each vector x_i
/// * `alpha` - Scalar
/// * `x` - Device array of device pointers to each vector x_i
/// * `incx` - Stride between consecutive elements of each x_i
/// * `batch_count` - Number of instances in the batch
pub fn scal_batched<T>(
    handle: &Handle,
    n: i32,
    alpha: &T,
    x: *const *mut T,
    incx: i32,
    batch_count: i32,
) -> Result<()>
where
    T: ScalBatchedType,
{
    T::rocblas_scal_batched(handle, n, alpha, x, incx, batch_count)
}

/// Scale vectors in a strided batch by a scalar
/// 
/// x_i := alpha * x_i, for i = 1,...,batch_count
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in each vector x_i
/// * `alpha` - Scalar
/// * `x` - Device pointer to first vector x_1
/// * `incx` - Stride between consecutive elements of each x_i
/// * `stride_x` - Stride from start of one vector (x_i) to the next (x_i+1)
/// * `batch_count` - Number of instances in the batch
pub fn scal_strided_batched<T>(
    handle: &Handle,
    n: i32,
    alpha: &T,
    x: *mut T,
    incx: i32,
    stride_x: i64,
    batch_count: i32,
) -> Result<()>
where
    T: ScalStridedBatchedType,
{
    T::rocblas_scal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
}

//==============================================================================
// COPY functions
//==============================================================================

/// Copy a vector
/// 
/// y := x
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in vectors x and y
/// * `x` - Device pointer to vector x
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Device pointer to vector y
/// * `incy` - Stride between consecutive elements of y
pub fn copy<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    y: *mut T,
    incy: i32,
) -> Result<()>
where
    T: CopyType,
{
    T::rocblas_copy(handle, n, x, incx, y, incy)
}

/// Copy vectors in a batch
/// 
/// y_i := x_i, for i = 1,...,batch_count
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in each vector x_i and y_i
/// * `x` - Device array of device pointers to each vector x_i
/// * `incx` - Stride between consecutive elements of each x_i
/// * `y` - Device array of device pointers to each vector y_i
/// * `incy` - Stride between consecutive elements of each y_i
/// * `batch_count` - Number of instances in the batch
pub fn copy_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const *const T,
    incx: i32,
    y: *const *mut T,
    incy: i32,
    batch_count: i32,
) -> Result<()>
where
    T: CopyBatchedType,
{
    T::rocblas_copy_batched(handle, n, x, incx, y, incy, batch_count)
}

/// Copy vectors in a strided batch
/// 
/// y_i := x_i, for i = 1,...,batch_count
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in each vector x_i and y_i
/// * `x` - Device pointer to first vector x_1
/// * `incx` - Stride between consecutive elements of each x_i
/// * `stridex` - Stride from start of one vector (x_i) to the next (x_i+1)
/// * `y` - Device pointer to first vector y_1
/// * `incy` - Stride between consecutive elements of each y_i
/// * `stridey` - Stride from start of one vector (y_i) to the next (y_i+1)
/// * `batch_count` - Number of instances in the batch
pub fn copy_strided_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    stridex: i64,
    y: *mut T,
    incy: i32,
    stridey: i64,
    batch_count: i32,
) -> Result<()>
where
    T: CopyStridedBatchedType,
{
    T::rocblas_copy_strided_batched(handle, n, x, incx, stridex, y, incy, stridey, batch_count)
}

//==============================================================================
// DOT functions
//==============================================================================

/// Compute the dot product of two vectors
/// 
/// result := x * y
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in vectors x and y
/// * `x` - Device pointer to vector x
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Device pointer to vector y
/// * `incy` - Stride between consecutive elements of y
/// * `result` - Pointer to the result
pub fn dot<T, R>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    y: *const T,
    incy: i32,
    result: *mut R,
) -> Result<()>
where
    T: DotType<ResultType = R>,
{
    T::rocblas_dot(handle, n, x, incx, y, incy, result)
}

/// Compute the dot product of two complex vectors
/// 
/// result := x * y (non-conjugated dot product)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in vectors x and y
/// * `x` - Device pointer to vector x
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Device pointer to vector y
/// * `incy` - Stride between consecutive elements of y
/// * `result` - Pointer to the result
pub fn dotu<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    y: *const T,
    incy: i32,
    result: *mut T,
) -> Result<()>
where
    T: DotuType,
{
    T::rocblas_dotu(handle, n, x, incx, y, incy, result)
}

/// Compute the conjugated dot product of two complex vectors
/// 
/// result := conjugate(x) * y
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in vectors x and y
/// * `x` - Device pointer to vector x
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Device pointer to vector y
/// * `incy` - Stride between consecutive elements of y
/// * `result` - Pointer to the result
pub fn dotc<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    y: *const T,
    incy: i32,
    result: *mut T,
) -> Result<()>
where
    T: DotcType,
{
    T::rocblas_dotc(handle, n, x, incx, y, incy, result)
}

//==============================================================================
// Type traits for implementation
//==============================================================================

/// Trait for types that can be used with scal
pub trait ScalType {
    fn rocblas_scal(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *mut Self,
        incx: i32,
    ) -> Result<()>;
}

impl ScalType for f32 {
    fn rocblas_scal(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *mut Self,
        incx: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sscal(handle.as_raw(), n, alpha, x, incx)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl ScalType for f64 {
    fn rocblas_scal(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *mut Self,
        incx: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dscal(handle.as_raw(), n, alpha, x, incx)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl ScalType for ffi::rocblas_float_complex {
    fn rocblas_scal(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *mut Self,
        incx: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cscal(handle.as_raw(), n, alpha, x, incx)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl ScalType for ffi::rocblas_double_complex {
    fn rocblas_scal(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *mut Self,
        incx: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zscal(handle.as_raw(), n, alpha, x, incx)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with scal_batched
pub trait ScalBatchedType {
    fn rocblas_scal_batched(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *const *mut Self,
        incx: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl ScalBatchedType for f32 {
    fn rocblas_scal_batched(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *const *mut Self,
        incx: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sscal_batched(handle.as_raw(), n, alpha, x, incx, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl ScalBatchedType for f64 {
    fn rocblas_scal_batched(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *const *mut Self,
        incx: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dscal_batched(handle.as_raw(), n, alpha, x, incx, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl ScalBatchedType for ffi::rocblas_float_complex {
    fn rocblas_scal_batched(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *const *mut Self,
        incx: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cscal_batched(handle.as_raw(), n, alpha, x, incx, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl ScalBatchedType for ffi::rocblas_double_complex {
    fn rocblas_scal_batched(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *const *mut Self,
        incx: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zscal_batched(handle.as_raw(), n, alpha, x, incx, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with scal_strided_batched
pub trait ScalStridedBatchedType {
    fn rocblas_scal_strided_batched(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *mut Self,
        incx: i32,
        stride_x: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl ScalStridedBatchedType for f32 {
    fn rocblas_scal_strided_batched(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *mut Self,
        incx: i32,
        stride_x: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sscal_strided_batched(handle.as_raw(), n, alpha, x, incx, stride_x, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl ScalStridedBatchedType for f64 {
    fn rocblas_scal_strided_batched(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *mut Self,
        incx: i32,
        stride_x: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dscal_strided_batched(handle.as_raw(), n, alpha, x, incx, stride_x, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl ScalStridedBatchedType for ffi::rocblas_float_complex {
    fn rocblas_scal_strided_batched(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *mut Self,
        incx: i32,
        stride_x: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cscal_strided_batched(handle.as_raw(), n, alpha, x, incx, stride_x, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl ScalStridedBatchedType for ffi::rocblas_double_complex {
    fn rocblas_scal_strided_batched(
        handle: &Handle,
        n: i32,
        alpha: &Self,
        x: *mut Self,
        incx: i32,
        stride_x: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zscal_strided_batched(handle.as_raw(), n, alpha, x, incx, stride_x, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with copy
pub trait CopyType {
    fn rocblas_copy(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *mut Self,
        incy: i32,
    ) -> Result<()>;
}

impl CopyType for f32 {
    fn rocblas_copy(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_scopy(handle.as_raw(), n, x, incx, y, incy)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl CopyType for f64 {
    fn rocblas_copy(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dcopy(handle.as_raw(), n, x, incx, y, incy)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl CopyType for ffi::rocblas_float_complex {
    fn rocblas_copy(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ccopy(handle.as_raw(), n, x, incx, y, incy)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl CopyType for ffi::rocblas_double_complex {
    fn rocblas_copy(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *mut Self,
        incy: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zcopy(handle.as_raw(), n, x, incx, y, incy)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with copy_batched
pub trait CopyBatchedType {
    fn rocblas_copy_batched(
        handle: &Handle,
        n: i32,
        x: *const *const Self,
        incx: i32,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl CopyBatchedType for f32 {
    fn rocblas_copy_batched(
        handle: &Handle,
        n: i32,
        x: *const *const Self,
        incx: i32,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_scopy_batched(handle.as_raw(), n, x, incx, y, incy, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl CopyBatchedType for f64 {
    fn rocblas_copy_batched(
        handle: &Handle,
        n: i32,
        x: *const *const Self,
        incx: i32,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dcopy_batched(handle.as_raw(), n, x, incx, y, incy, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl CopyBatchedType for ffi::rocblas_float_complex {
    fn rocblas_copy_batched(
        handle: &Handle,
        n: i32,
        x: *const *const Self,
        incx: i32,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ccopy_batched(handle.as_raw(), n, x, incx, y, incy, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl CopyBatchedType for ffi::rocblas_double_complex {
    fn rocblas_copy_batched(
        handle: &Handle,
        n: i32,
        x: *const *const Self,
        incx: i32,
        y: *const *mut Self,
        incy: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zcopy_batched(handle.as_raw(), n, x, incx, y, incy, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with copy_strided_batched
pub trait CopyStridedBatchedType {
    fn rocblas_copy_strided_batched(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        stridex: i64,
        y: *mut Self,
        incy: i32,
        stridey: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl CopyStridedBatchedType for f32 {
    fn rocblas_copy_strided_batched(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        stridex: i64,
        y: *mut Self,
        incy: i32,
        stridey: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_scopy_strided_batched(handle.as_raw(), n, x, incx, stridex, y, incy, stridey, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl CopyStridedBatchedType for f64 {
    fn rocblas_copy_strided_batched(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        stridex: i64,
        y: *mut Self,
        incy: i32,
        stridey: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dcopy_strided_batched(handle.as_raw(), n, x, incx, stridex, y, incy, stridey, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl CopyStridedBatchedType for ffi::rocblas_float_complex {
    fn rocblas_copy_strided_batched(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        stridex: i64,
        y: *mut Self,
        incy: i32,
        stridey: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ccopy_strided_batched(handle.as_raw(), n, x, incx, stridex, y, incy, stridey, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl CopyStridedBatchedType for ffi::rocblas_double_complex {
    fn rocblas_copy_strided_batched(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        stridex: i64,
        y: *mut Self,
        incy: i32,
        stridey: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zcopy_strided_batched(handle.as_raw(), n, x, incx, stridex, y, incy, stridey, batch_count)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with dot
pub trait DotType {
    type ResultType;

    fn rocblas_dot(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        result: *mut Self::ResultType,
    ) -> Result<()>;
}

impl DotType for f32 {
    type ResultType = f32;

    fn rocblas_dot(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        result: *mut Self::ResultType,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sdot(handle.as_raw(), n, x, incx, y, incy, result)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl DotType for f64 {
    type ResultType = f64;

    fn rocblas_dot(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        result: *mut Self::ResultType,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ddot(handle.as_raw(), n, x, incx, y, incy, result)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl DotType for ffi::rocblas_half {
    type ResultType = ffi::rocblas_half;

    fn rocblas_dot(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        result: *mut Self::ResultType,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_hdot(handle.as_raw(), n, x, incx, y, incy, result)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl DotType for ffi::rocblas_bfloat16 {
    type ResultType = ffi::rocblas_bfloat16;

    fn rocblas_dot(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        result: *mut Self::ResultType,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_bfdot(handle.as_raw(), n, x, incx, y, incy, result)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with dotu
pub trait DotuType {
    fn rocblas_dotu(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        result: *mut Self,
    ) -> Result<()>;
}

impl DotuType for ffi::rocblas_float_complex {
    fn rocblas_dotu(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        result: *mut Self,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cdotu(handle.as_raw(), n, x, incx, y, incy, result)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl DotuType for ffi::rocblas_double_complex {
    fn rocblas_dotu(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        result: *mut Self,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zdotu(handle.as_raw(), n, x, incx, y, incy, result)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with dotc
pub trait DotcType {
    fn rocblas_dotc(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        result: *mut Self,
    ) -> Result<()>;
}

impl DotcType for ffi::rocblas_float_complex {
    fn rocblas_dotc(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        result: *mut Self,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cdotc(handle.as_raw(), n, x, incx, y, incy, result)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl DotcType for ffi::rocblas_double_complex {
    fn rocblas_dotc(
        handle: &Handle,
        n: i32,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        result: *mut Self,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zdotc(handle.as_raw(), n, x, incx, y, incy, result)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Add a placeholder declaration for the remaining functions
// that we haven't fully implemented yet

// BLAS Level 1
pub fn axpy<T>(_handle: &Handle, _n: i32, _alpha: &T, _x: *const T, _incx: i32, _y: *mut T, _incy: i32) -> Result<()> { todo!() }
pub fn nrm2<T, R>(_handle: &Handle, _n: i32, _x: *const T, _incx: i32, _result: *mut R) -> Result<()> { todo!() }
pub fn asum<T, R>(_handle: &Handle, _n: i32, _x: *const T, _incx: i32, _result: *mut R) -> Result<()> { todo!() }
pub fn amax<T>(_handle: &Handle, _n: i32, _x: *const T, _incx: i32, _result: *mut i32) -> Result<()> { todo!() }
pub fn amin<T>(_handle: &Handle, _n: i32, _x: *const T, _incx: i32, _result: *mut i32) -> Result<()> { todo!() }
pub fn swap<T>(_handle: &Handle, _n: i32, _x: *mut T, _incx: i32, _y: *mut T, _incy: i32) -> Result<()> { todo!() }
pub fn rot<T>(_handle: &Handle, _n: i32, _x: *mut T, _incx: i32, _y: *mut T, _incy: i32, _c: *const f32, _s: *const f32) -> Result<()> { todo!() }
pub fn rotg<T>(_handle: &Handle, _a: *mut T, _b: *mut T, _c: *mut T, _s: *mut T) -> Result<()> { todo!() }
pub fn rotm<T>(_handle: &Handle, _n: i32, _x: *mut T, _incx: i32, _y: *mut T, _incy: i32, _param: *const T) -> Result<()> { todo!() }
pub fn rotmg<T>(_handle: &Handle, _d1: *mut T, _d2: *mut T, _x1: *mut T, _y1: *const T, _param: *mut T) -> Result<()> { todo!() }

// BLAS Level 1 - Batched
pub fn axpy_batched<T>(_handle: &Handle, _n: i32, _alpha: &T, _x: *const *const T, _incx: i32, _y: *const *mut T, _incy: i32, _batch_count: i32) -> Result<()> { todo!() }
pub fn dot_batched<T, R>(_handle: &Handle, _n: i32, _x: *const *const T, _incx: i32, _y: *const *const T, _incy: i32, _batch_count: i32, _result: *mut R) -> Result<()> { todo!() }
pub fn dotu_batched<T>(_handle: &Handle, _n: i32, _x: *const *const T, _incx: i32, _y: *const *const T, _incy: i32, _batch_count: i32, _result: *mut T) -> Result<()> { todo!() }
pub fn dotc_batched<T>(_handle: &Handle, _n: i32, _x: *const *const T, _incx: i32, _y: *const *const T, _incy: i32, _batch_count: i32, _result: *mut T) -> Result<()> { todo!() }
pub fn nrm2_batched<T, R>(_handle: &Handle, _n: i32, _x: *const *const T, _incx: i32, _batch_count: i32, _result: *mut R) -> Result<()> { todo!() }
pub fn asum_batched<T, R>(_handle: &Handle, _n: i32, _x: *const *const T, _incx: i32, _batch_count: i32, _result: *mut R) -> Result<()> { todo!() }
pub fn amax_batched<T>(_handle: &Handle, _n: i32, _x: *const *const T, _incx: i32, _batch_count: i32, _result: *mut i32) -> Result<()> { todo!() }
pub fn amin_batched<T>(_handle: &Handle, _n: i32, _x: *const *const T, _incx: i32, _batch_count: i32, _result: *mut i32) -> Result<()> { todo!() }
pub fn swap_batched<T>(_handle: &Handle, _n: i32, _x: *const *mut T, _incx: i32, _y: *const *mut T, _incy: i32, _batch_count: i32) -> Result<()> { todo!() }
pub fn rot_batched<T>(_handle: &Handle, _n: i32, _x: *const *mut T, _incx: i32, _y: *const *mut T, _incy: i32, _c: *const f32, _s: *const f32, _batch_count: i32) -> Result<()> { todo!() }
pub fn rotg_batched<T>(_handle: &Handle, _a: *const *mut T, _b: *const *mut T, _c: *const *mut T, _s: *const *mut T, _batch_count: i32) -> Result<()> { todo!() }
pub fn rotm_batched<T>(_handle: &Handle, _n: i32, _x: *const *mut T, _incx: i32, _y: *const *mut T, _incy: i32, _param: *const *const T, _batch_count: i32) -> Result<()> { todo!() }
pub fn rotmg_batched<T>(_handle: &Handle, _d1: *const *mut T, _d2: *const *mut T, _x1: *const *mut T, _y1: *const *const T, _param: *const *mut T, _batch_count: i32) -> Result<()> { todo!() }

// BLAS Level 1 - Strided Batched
pub fn axpy_strided_batched<T>(_handle: &Handle, _n: i32, _alpha: &T, _x: *const T, _incx: i32, _stridex: i64, _y: *mut T, _incy: i32, _stridey: i64, _batch_count: i32) -> Result<()> { todo!() }
pub fn dot_strided_batched<T, R>(_handle: &Handle, _n: i32, _x: *const T, _incx: i32, _stridex: i64, _y: *const T, _incy: i32, _stridey: i64, _batch_count: i32, _result: *mut R) -> Result<()> { todo!() }
pub fn dotu_strided_batched<T>(_handle: &Handle, _n: i32, _x: *const T, _incx: i32, _stridex: i64, _y: *const T, _incy: i32, _stridey: i64, _batch_count: i32, _result: *mut T) -> Result<()> { todo!() }
pub fn dotc_strided_batched<T>(_handle: &Handle, _n: i32, _x: *const T, _incx: i32, _stridex: i64, _y: *const T, _incy: i32, _stridey: i64, _batch_count: i32, _result: *mut T) -> Result<()> { todo!() }
pub fn nrm2_strided_batched<T, R>(_handle: &Handle, _n: i32, _x: *const T, _incx: i32, _stridex: i64, _batch_count: i32, _result: *mut R) -> Result<()> { todo!() }
pub fn asum_strided_batched<T, R>(_handle: &Handle, _n: i32, _x: *const T, _incx: i32, _stridex: i64, _batch_count: i32, _result: *mut R) -> Result<()> { todo!() }
pub fn amax_strided_batched<T>(_handle: &Handle, _n: i32, _x: *const T, _incx: i32, _stridex: i64, _batch_count: i32, _result: *mut i32) -> Result<()> { todo!() }
pub fn amin_strided_batched<T>(_handle: &Handle, _n: i32, _x: *const T, _incx: i32, _stridex: i64, _batch_count: i32, _result: *mut i32) -> Result<()> { todo!() }
pub fn swap_strided_batched<T>(_handle: &Handle, _n: i32, _x: *mut T, _incx: i32, _stridex: i64, _y: *mut T, _incy: i32, _stridey: i64, _batch_count: i32) -> Result<()> { todo!() }
pub fn rot_strided_batched<T>(_handle: &Handle, _n: i32, _x: *mut T, _incx: i32, _stridex: i64, _y: *mut T, _incy: i32, _stridey: i64, _c: *const f32, _s: *const f32, _batch_count: i32) -> Result<()> { todo!() }
pub fn rotg_strided_batched<T>(_handle: &Handle, _a: *mut T, _stridea: i64, _b: *mut T, _strideb: i64, _c: *mut T, _stridec: i64, _s: *mut T, _strides: i64, _batch_count: i32) -> Result<()> { todo!() }
pub fn rotm_strided_batched<T>(_handle: &Handle, _n: i32, _x: *mut T, _incx: i32, _stridex: i64, _y: *mut T, _incy: i32, _stridey: i64, _param: *const T, _param_stride: i64, _batch_count: i32) -> Result<()> { todo!() }
pub fn rotmg_strided_batched<T>(_handle: &Handle, _d1: *mut T, _stride_d1: i64, _d2: *mut T, _stride_d2: i64, _x1: *mut T, _stride_x1: i64, _y1: *const T, _stride_y1: i64, _param: *mut T, _stride_param: i64, _batch_count: i32) -> Result<()> { todo!() }