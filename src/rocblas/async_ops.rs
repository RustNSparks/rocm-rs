// src/rocblas/async_ops.rs
use std::mem;
use crate::rocblas::ffi;
use crate::rocblas::error::{Error, Result};
use crate::rocblas::handle::Handle;
use crate::hip::Stream;
use crate::rocblas::bindings::hipEvent_t;

/// Set vector asynchronously from host to device
///
/// # Arguments
/// * `n` - Number of elements in the vector
/// * `x` - Source vector on the host
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Destination vector on the device
/// * `incy` - Stride between consecutive elements of y
/// * `stream` - HIP stream to use for the transfer
pub fn set_vector_async<T: Copy>(
    n: i32,
    x: &[T],
    incx: i32,
    y: *mut T,
    incy: i32,
    stream: &Stream,
) -> Result<()> {
    let elem_size = mem::size_of::<T>() as i32;

    let status = unsafe {
        ffi::rocblas_set_vector_async(
            n,
            elem_size,
            x.as_ptr() as *const std::os::raw::c_void,
            incx,
            y as *mut std::os::raw::c_void,
            incy,
            stream.as_raw() as crate::rocblas::bindings::hipStream_t,
        )
    };

    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Set vector asynchronously from host to device (64-bit version)
///
/// # Arguments
/// * `n` - Number of elements in the vector
/// * `x` - Source vector on the host
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Destination vector on the device
/// * `incy` - Stride between consecutive elements of y
/// * `stream` - HIP stream to use for the transfer
pub fn set_vector_async_64<T: Copy>(
    n: i64,
    x: &[T],
    incx: i64,
    y: *mut T,
    incy: i64,
    stream: &Stream,
) -> Result<()> {
    let elem_size = mem::size_of::<T>() as i64;

    let status = unsafe {
        ffi::rocblas_set_vector_async_64(
            n,
            elem_size,
            x.as_ptr() as *const std::os::raw::c_void,
            incx,
            y as *mut std::os::raw::c_void,
            incy,
            stream.as_raw() as crate::rocblas::bindings::hipStream_t,
        )
    };

    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Get vector asynchronously from device to host
///
/// # Arguments
/// * `n` - Number of elements in the vector
/// * `x` - Source vector on the device
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Destination vector on the host
/// * `incy` - Stride between consecutive elements of y
/// * `stream` - HIP stream to use for the transfer
pub fn get_vector_async<T: Copy>(
    n: i32,
    x: *const T,
    incx: i32,
    y: &mut [T],
    incy: i32,
    stream: &Stream,
) -> Result<()> {
    let elem_size = mem::size_of::<T>() as i32;

    let status = unsafe {
        ffi::rocblas_get_vector_async(
            n,
            elem_size,
            x as *const std::os::raw::c_void,
            incx,
            y.as_mut_ptr() as *mut std::os::raw::c_void,
            incy,
            stream.as_raw() as crate::rocblas::bindings::hipStream_t,
        )
    };

    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Get vector asynchronously from device to host (64-bit version)
///
/// # Arguments
/// * `n` - Number of elements in the vector
/// * `x` - Source vector on the device
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Destination vector on the host
/// * `incy` - Stride between consecutive elements of y
/// * `stream` - HIP stream to use for the transfer
pub fn get_vector_async_64<T: Copy>(
    n: i64,
    x: *const T,
    incx: i64,
    y: &mut [T],
    incy: i64,
    stream: &Stream,
) -> Result<()> {
    let elem_size = mem::size_of::<T>() as i64;

    let status = unsafe {
        ffi::rocblas_get_vector_async_64(
            n,
            elem_size,
            x as *const std::os::raw::c_void,
            incx,
            y.as_mut_ptr() as *mut std::os::raw::c_void,
            incy,
            stream.as_raw() as crate::rocblas::bindings::hipStream_t,
        )
    };

    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Set matrix asynchronously from host to device
///
/// # Arguments
/// * `rows` - Number of rows of the matrix
/// * `cols` - Number of columns of the matrix
/// * `a` - Source matrix on the host
/// * `lda` - Leading dimension of a
/// * `b` - Destination matrix on the device
/// * `ldb` - Leading dimension of b
/// * `stream` - HIP stream to use for the transfer
pub fn set_matrix_async<T: Copy>(
    rows: i32,
    cols: i32,
    a: &[T],
    lda: i32,
    b: *mut T,
    ldb: i32,
    stream: &Stream,
) -> Result<()> {
    let elem_size = mem::size_of::<T>() as i32;

    let status = unsafe {
        ffi::rocblas_set_matrix_async(
            rows,
            cols,
            elem_size,
            a.as_ptr() as *const std::os::raw::c_void,
            lda,
            b as *mut std::os::raw::c_void,
            ldb,
            stream.as_raw() as crate::rocblas::bindings::hipStream_t,
        )
    };

    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Set matrix asynchronously from host to device (64-bit version)
///
/// # Arguments
/// * `rows` - Number of rows of the matrix
/// * `cols` - Number of columns of the matrix
/// * `a` - Source matrix on the host
/// * `lda` - Leading dimension of a
/// * `b` - Destination matrix on the device
/// * `ldb` - Leading dimension of b
/// * `stream` - HIP stream to use for the transfer
pub fn set_matrix_async_64<T: Copy>(
    rows: i64,
    cols: i64,
    a: &[T],
    lda: i64,
    b: *mut T,
    ldb: i64,
    stream: &Stream,
) -> Result<()> {
    let elem_size = mem::size_of::<T>() as i64;

    let status = unsafe {
        ffi::rocblas_set_matrix_async_64(
            rows,
            cols,
            elem_size,
            a.as_ptr() as *const std::os::raw::c_void,
            lda,
            b as *mut std::os::raw::c_void,
            ldb,
            stream.as_raw() as crate::rocblas::bindings::hipStream_t,
        )
    };

    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Get matrix asynchronously from device to host
///
/// # Arguments
/// * `rows` - Number of rows of the matrix
/// * `cols` - Number of columns of the matrix
/// * `a` - Source matrix on the device
/// * `lda` - Leading dimension of a
/// * `b` - Destination matrix on the host
/// * `ldb` - Leading dimension of b
/// * `stream` - HIP stream to use for the transfer
pub fn get_matrix_async<T: Copy>(
    rows: i32,
    cols: i32,
    a: *const T,
    lda: i32,
    b: &mut [T],
    ldb: i32,
    stream: &Stream,
) -> Result<()> {
    let elem_size = mem::size_of::<T>() as i32;

    let status = unsafe {
        ffi::rocblas_get_matrix_async(
            rows,
            cols,
            elem_size,
            a as *const std::os::raw::c_void,
            lda,
            b.as_mut_ptr() as *mut std::os::raw::c_void,
            ldb,
            stream.as_raw() as crate::rocblas::bindings::hipStream_t,
        )
    };

    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Get matrix asynchronously from device to host (64-bit version)
///
/// # Arguments
/// * `rows` - Number of rows of the matrix
/// * `cols` - Number of columns of the matrix
/// * `a` - Source matrix on the device
/// * `lda` - Leading dimension of a
/// * `b` - Destination matrix on the host
/// * `ldb` - Leading dimension of b
/// * `stream` - HIP stream to use for the transfer
pub fn get_matrix_async_64<T: Copy>(
    rows: i64,
    cols: i64,
    a: *const T,
    lda: i64,
    b: &mut [T],
    ldb: i64,
    stream: &Stream,
) -> Result<()> {
    let elem_size = mem::size_of::<T>() as i64;

    let status = unsafe {
        ffi::rocblas_get_matrix_async_64(
            rows,
            cols,
            elem_size,
            a as *const std::os::raw::c_void,
            lda,
            b.as_mut_ptr() as *mut std::os::raw::c_void,
            ldb,
            stream.as_raw() as crate::rocblas::bindings::hipStream_t,
        )
    };

    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Set start and stop events for RocBLAS operations
///
/// This function allows performance measurement of RocBLAS operations
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `start_event` - HIP event to record at the start of operations
/// * `stop_event` - HIP event to record at the end of operations
pub fn set_start_stop_events(
    handle: &Handle,
    start_event: hipEvent_t,
    stop_event: hipEvent_t,
) -> Result<()> {
    let status = unsafe {
        ffi::rocblas_set_start_stop_events(
            handle.as_raw(),
            start_event,
            stop_event,
        )
    };

    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }

    Ok(())
}