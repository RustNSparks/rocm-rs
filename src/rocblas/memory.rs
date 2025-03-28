/*!
# Memory Transfer Utilities

This module provides utilities for transferring data between host and device memory
for vectors and matrices.
*/

use std::os::raw::c_void;
use crate::rocblas::error::{Result, check_error};
use crate::rocblas::ffi;
use crate::rocblas::Handle;
use crate::hip::ffi::hipStream_t;

/// Transfer a vector from host to device
///
/// # Arguments
///
/// * `n` - Number of elements in the vector
/// * `elem_size` - Size in bytes of each element
/// * `x` - Host pointer to the source data
/// * `incx` - Stride between consecutive elements in x
/// * `y` - Device pointer to the destination
/// * `incy` - Stride between consecutive elements in y
///
/// # Returns
///
/// A result indicating success or an error
pub fn set_vector(
    n: i32,
    elem_size: i32,
    x: *const c_void,
    incx: i32,
    y: *mut c_void,
    incy: i32,
) -> Result<()> {
    unsafe {
        check_error(ffi::rocblas_set_vector(n, elem_size, x, incx, y, incy))
    }
}

/// Transfer a vector from host to device using 64-bit indexing
///
/// # Arguments
///
/// * `n` - Number of elements in the vector
/// * `elem_size` - Size in bytes of each element
/// * `x` - Host pointer to the source data
/// * `incx` - Stride between consecutive elements in x
/// * `y` - Device pointer to the destination
/// * `incy` - Stride between consecutive elements in y
///
/// # Returns
///
/// A result indicating success or an error
pub fn set_vector_64(
    n: i64,
    elem_size: i64,
    x: *const c_void,
    incx: i64,
    y: *mut c_void,
    incy: i64,
) -> Result<()> {
    unsafe {
        check_error(ffi::rocblas_set_vector_64(n, elem_size, x, incx, y, incy))
    }
}

/// Transfer a vector from device to host
///
/// # Arguments
///
/// * `n` - Number of elements in the vector
/// * `elem_size` - Size in bytes of each element
/// * `x` - Device pointer to the source data
/// * `incx` - Stride between consecutive elements in x
/// * `y` - Host pointer to the destination
/// * `incy` - Stride between consecutive elements in y
///
/// # Returns
///
/// A result indicating success or an error
pub fn get_vector(
    n: i32,
    elem_size: i32,
    x: *const c_void,
    incx: i32,
    y: *mut c_void,
    incy: i32,
) -> Result<()> {
    unsafe {
        check_error(ffi::rocblas_get_vector(n, elem_size, x, incx, y, incy))
    }
}

/// Transfer a vector from device to host using 64-bit indexing
///
/// # Arguments
///
/// * `n` - Number of elements in the vector
/// * `elem_size` - Size in bytes of each element
/// * `x` - Device pointer to the source data
/// * `incx` - Stride between consecutive elements in x
/// * `y` - Host pointer to the destination
/// * `incy` - Stride between consecutive elements in y
///
/// # Returns
///
/// A result indicating success or an error
pub fn get_vector_64(
    n: i64,
    elem_size: i64,
    x: *const c_void,
    incx: i64,
    y: *mut c_void,
    incy: i64,
) -> Result<()> {
    unsafe {
        check_error(ffi::rocblas_get_vector_64(n, elem_size, x, incx, y, incy))
    }
}

/// Transfer a matrix from host to device
///
/// # Arguments
///
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix
/// * `elem_size` - Size in bytes of each element
/// * `a` - Host pointer to the source matrix
/// * `lda` - Leading dimension of the source matrix
/// * `b` - Device pointer to the destination matrix
/// * `ldb` - Leading dimension of the destination matrix
///
/// # Returns
///
/// A result indicating success or an error
pub fn set_matrix(
    rows: i32,
    cols: i32,
    elem_size: i32,
    a: *const c_void,
    lda: i32,
    b: *mut c_void,
    ldb: i32,
) -> Result<()> {
    unsafe {
        check_error(ffi::rocblas_set_matrix(rows, cols, elem_size, a, lda, b, ldb))
    }
}

/// Transfer a matrix from host to device using 64-bit indexing
///
/// # Arguments
///
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix
/// * `elem_size` - Size in bytes of each element
/// * `a` - Host pointer to the source matrix
/// * `lda` - Leading dimension of the source matrix
/// * `b` - Device pointer to the destination matrix
/// * `ldb` - Leading dimension of the destination matrix
///
/// # Returns
///
/// A result indicating success or an error
pub fn set_matrix_64(
    rows: i64,
    cols: i64,
    elem_size: i64,
    a: *const c_void,
    lda: i64,
    b: *mut c_void,
    ldb: i64,
) -> Result<()> {
    unsafe {
        check_error(ffi::rocblas_set_matrix_64(rows, cols, elem_size, a, lda, b, ldb))
    }
}

/// Transfer a matrix from device to host
///
/// # Arguments
///
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix
/// * `elem_size` - Size in bytes of each element
/// * `a` - Device pointer to the source matrix
/// * `lda` - Leading dimension of the source matrix
/// * `b` - Host pointer to the destination matrix
/// * `ldb` - Leading dimension of the destination matrix
///
/// # Returns
///
/// A result indicating success or an error
pub fn get_matrix(
    rows: i32,
    cols: i32,
    elem_size: i32,
    a: *const c_void,
    lda: i32,
    b: *mut c_void,
    ldb: i32,
) -> Result<()> {
    unsafe {
        check_error(ffi::rocblas_get_matrix(rows, cols, elem_size, a, lda, b, ldb))
    }
}

/// Transfer a matrix from device to host using 64-bit indexing
///
/// # Arguments
///
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix
/// * `elem_size` - Size in bytes of each element
/// * `a` - Device pointer to the source matrix
/// * `lda` - Leading dimension of the source matrix
/// * `b` - Host pointer to the destination matrix
/// * `ldb` - Leading dimension of the destination matrix
///
/// # Returns
///
/// A result indicating success or an error
pub fn get_matrix_64(
    rows: i64,
    cols: i64,
    elem_size: i64,
    a: *const c_void,
    lda: i64,
    b: *mut c_void,
    ldb: i64,
) -> Result<()> {
    unsafe {
        check_error(ffi::rocblas_get_matrix_64(rows, cols, elem_size, a, lda, b, ldb))
    }
}

/// Asynchronously transfer a vector from host to device
///
/// # Arguments
///
/// * `n` - Number of elements in the vector
/// * `elem_size` - Size in bytes of each element
/// * `x` - Host pointer to the source data
/// * `incx` - Stride between consecutive elements in x
/// * `y` - Device pointer to the destination
/// * `incy` - Stride between consecutive elements in y
/// * `stream` - Stream to use for the transfer
///
/// # Returns
///
/// A result indicating success or an error
pub fn set_vector_async(
    n: i32,
    elem_size: i32,
    x: *const c_void,
    incx: i32,
    y: *mut c_void,
    incy: i32,
    stream: hipStream_t,
) -> Result<()> {
    unsafe {
        check_error(ffi::rocblas_set_vector_async(n, elem_size, x, incx, y, incy, stream as *mut _))
    }
}

/// Asynchronously transfer a vector from device to host
///
/// # Arguments
///
/// * `n` - Number of elements in the vector
/// * `elem_size` - Size in bytes of each element
/// * `x` - Device pointer to the source data
/// * `incx` - Stride between consecutive elements in x
/// * `y` - Host pointer to the destination
/// * `incy` - Stride between consecutive elements in y
/// * `stream` - Stream to use for the transfer
///
/// # Returns
///
/// A result indicating success or an error
pub fn get_vector_async(
    n: i32,
    elem_size: i32,
    x: *const c_void,
    incx: i32,
    y: *mut c_void,
    incy: i32,
    stream: hipStream_t,
) -> Result<()> {
    unsafe {
        check_error(ffi::rocblas_get_vector_async(n, elem_size, x, incx, y, incy, stream as *mut _ ))
    }
}

/// Asynchronously transfer a matrix from host to device
///
/// # Arguments
///
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix
/// * `elem_size` - Size in bytes of each element
/// * `a` - Host pointer to the source matrix
/// * `lda` - Leading dimension of the source matrix
/// * `b` - Device pointer to the destination matrix
/// * `ldb` - Leading dimension of the destination matrix
/// * `stream` - Stream to use for the transfer
///
/// # Returns
///
/// A result indicating success or an error
pub fn set_matrix_async(
    rows: i32,
    cols: i32,
    elem_size: i32,
    a: *const c_void,
    lda: i32,
    b: *mut c_void,
    ldb: i32,
    stream: hipStream_t,
) -> Result<()> {
    unsafe {
        check_error(ffi::rocblas_set_matrix_async(rows, cols, elem_size, a, lda, b, ldb, stream as *mut _ ))
    }
}

/// Asynchronously transfer a matrix from device to host
///
/// # Arguments
///
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix
/// * `elem_size` - Size in bytes of each element
/// * `a` - Device pointer to the source matrix
/// * `lda` - Leading dimension of the source matrix
/// * `b` - Host pointer to the destination matrix
/// * `ldb` - Leading dimension of the destination matrix
/// * `stream` - Stream to use for the transfer
///
/// # Returns
///
/// A result indicating success or an error
pub fn get_matrix_async(
    rows: i32,
    cols: i32,
    elem_size: i32,
    a: *const c_void,
    lda: i32,
    b: *mut c_void,
    ldb: i32,
    stream: hipStream_t,
) -> Result<()> {
    unsafe {
        check_error(ffi::rocblas_get_matrix_async(rows, cols, elem_size, a, lda, b, ldb, stream as *mut _))
    }
}