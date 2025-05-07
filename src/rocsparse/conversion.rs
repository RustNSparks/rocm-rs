//! Matrix format conversion utilities

use std::ffi::c_void;
use crate::rocsparse::descriptor::{IndexBase, MatrixDescriptor};
use crate::rocsparse::handle::Handle;
use crate::rocsparse::{rocsparse_action__rocsparse_action_numeric, rocsparse_action__rocsparse_action_symbolic, rocsparse_create_identity_permutation, rocsparse_csr2csc_buffer_size, rocsparse_csrsort, rocsparse_csrsort_buffer_size, rocsparse_scsr2csc};
use crate::rocsparse::error::status_to_result;
use crate::rocsparse::error::*;

/// Convert CSR to CSC (Compressed Sparse Column) format
pub fn csr_to_csc<T: Copy + 'static>(
    handle: &Handle,
    m: i32,
    n: i32,
    nnz: i32,
    csr_val: &[T],
    csr_row_ptr: &[i32],
    csr_col_ind: &[i32],
    csc_val: &mut [T],
    csc_row_ind: &mut [i32],
    csc_col_ptr: &mut [i32],
    copy_values: bool,
    idx_base: IndexBase,
) -> crate::rocsparse::error::Result<()> {
    // Get required buffer size
    let mut buffer_size = 0;
    let status = unsafe {
        rocsparse_csr2csc_buffer_size(
            handle.inner,
            m,
            n,
            nnz,
            csr_row_ptr.as_ptr(),
            csr_col_ind.as_ptr(),
            if copy_values {
                rocsparse_action__rocsparse_action_numeric
            } else {
                rocsparse_action__rocsparse_action_symbolic
            },
            &mut buffer_size,
        )
    };
    status_to_result(status)?;

    // Allocate temporary buffer
    let mut temp_buffer = vec![0u8; buffer_size];

    // Perform conversion based on type
    let status = convert_csr_to_csc(
        handle,
        m,
        n,
        nnz,
        csr_val,
        csr_row_ptr,
        csr_col_ind,
        csc_val,
        csc_row_ind,
        csc_col_ptr,
        copy_values,
        idx_base,
        temp_buffer.as_mut_ptr() as *mut c_void,
    );

    status
}

// Implementation for specific types
fn convert_csr_to_csc<T: 'static>(
    handle: &Handle,
    m: i32,
    n: i32,
    nnz: i32,
    csr_val: &[T],
    csr_row_ptr: &[i32],
    csr_col_ind: &[i32],
    csc_val: &mut [T],
    csc_row_ind: &mut [i32],
    csc_col_ptr: &mut [i32],
    copy_values: bool,
    idx_base: IndexBase,
    temp_buffer: *mut c_void,
) -> Result<()> {
    // This would need to be implemented for each supported type (f32, f64, complex, etc.)
    // For simplicity, I'm showing the f32 case only

    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let status = unsafe {
            rocsparse_scsr2csc(
                handle.inner,
                m,
                n,
                nnz,
                csr_val.as_ptr() as *const f32,
                csr_row_ptr.as_ptr(),
                csr_col_ind.as_ptr(),
                csc_val.as_mut_ptr() as *mut f32,
                csc_row_ind.as_mut_ptr(),
                csc_col_ptr.as_mut_ptr(),
                if copy_values {
                    rocsparse_action__rocsparse_action_numeric
                } else {
                    rocsparse_action__rocsparse_action_symbolic
                },
                idx_base.into(),
                temp_buffer,
            )
        };
        status_to_result(status)
    } else {
        Err(Error::NotImplemented)
    }
}

/// Create an identity permutation vector
pub fn create_identity_permutation(handle: &Handle, n: i32, p: &mut [i32]) -> Result<()> {
    let status = unsafe { rocsparse_create_identity_permutation(handle.inner, n, p.as_mut_ptr()) };
    status_to_result(status)
}

/// Sort a sparse CSR matrix
pub fn csr_sort(
    handle: &Handle,
    m: i32,
    n: i32,
    nnz: i32,
    descr: &MatrixDescriptor,
    csr_row_ptr: &[i32],
    csr_col_ind: &mut [i32],
    perm: Option<&mut [i32]>,
) -> Result<()> {
    // Get required buffer size
    let mut buffer_size = 0;
    let status = unsafe {
        rocsparse_csrsort_buffer_size(
            handle.inner,
            m,
            n,
            nnz,
            csr_row_ptr.as_ptr(),
            csr_col_ind.as_ptr(),
            &mut buffer_size,
        )
    };
    status_to_result(status)?;

    // Allocate temporary buffer
    let mut temp_buffer = vec![0u8; buffer_size];

    // Perform sort
    let status = unsafe { 
        rocsparse_csrsort(
            handle.inner,
            m,
            n,
            nnz,
            descr.inner,
            csr_row_ptr.as_ptr(),
            csr_col_ind.as_mut_ptr(),
            perm.map_or(std::ptr::null_mut(), |p| p.as_mut_ptr()),
            temp_buffer.as_mut_ptr() as *mut c_void,
        )
    };

    status_to_result(status)
}