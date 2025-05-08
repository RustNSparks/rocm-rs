//! Matrix pruning utilities

use std::ffi::c_void;
use crate::rocsparse::descriptor::MatrixDescriptor;
use crate::rocsparse::handle::Handle;
use crate::rocsparse::matrix::MatrixInfo;
use crate::rocsparse::error::{Result, status_to_result};
use crate::rocsparse::{
    rocsparse_sprune_csr2csr_nnz_by_percentage, rocsparse_dprune_csr2csr_nnz_by_percentage,
    rocsparse_sprune_csr2csr_by_percentage, rocsparse_dprune_csr2csr_by_percentage,
    rocsparse_sprune_dense2csr_buffer_size, rocsparse_dprune_dense2csr_buffer_size,
    rocsparse_sprune_dense2csr_nnz, rocsparse_dprune_dense2csr_nnz,
    rocsparse_sprune_dense2csr, rocsparse_dprune_dense2csr,
    rocsparse_sprune_dense2csr_by_percentage_buffer_size, rocsparse_dprune_dense2csr_by_percentage_buffer_size,
    rocsparse_sprune_dense2csr_nnz_by_percentage, rocsparse_dprune_dense2csr_nnz_by_percentage,
    rocsparse_sprune_dense2csr_by_percentage, rocsparse_dprune_dense2csr_by_percentage
};

/// Computes the number of non-zero elements per row and total non-zero elements
/// in a CSR matrix after pruning by percentage
pub fn prune_csr2csr_nnz_by_percentage<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    nnz_a: i32,
    csr_descr_a: &MatrixDescriptor,
    csr_val_a: &[T],
    csr_row_ptr_a: &[i32],
    csr_col_ind_a: &[i32],
    percentage: T,
    csr_descr_c: &MatrixDescriptor,
    csr_row_ptr_c: &mut [i32],
    info: &MatrixInfo,
    temp_buffer: *mut c_void,
) -> Result<i32>
where
    T: Copy + 'static,
{
    let mut nnz_total = 0;
    let status = prune_csr2csr_nnz_by_percentage_typed(
        handle,
        m,
        n,
        nnz_a,
        csr_descr_a,
        csr_val_a,
        csr_row_ptr_a,
        csr_col_ind_a,
        percentage,
        csr_descr_c,
        csr_row_ptr_c,
        &mut nnz_total,
        info,
        temp_buffer,
    );
    status.map(|_| nnz_total)
}

fn prune_csr2csr_nnz_by_percentage_typed<T: 'static>(
    handle: &Handle,
    m: i32,
    n: i32,
    nnz_a: i32,
    csr_descr_a: &MatrixDescriptor,
    csr_val_a: &[T],
    csr_row_ptr_a: &[i32],
    csr_col_ind_a: &[i32],
    percentage: T,
    csr_descr_c: &MatrixDescriptor,
    csr_row_ptr_c: &mut [i32],
    nnz_total: &mut i32,
    info: &MatrixInfo,
    temp_buffer: *mut c_void,
) -> Result<()> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let status = unsafe {
            rocsparse_sprune_csr2csr_nnz_by_percentage(
                handle.inner,
                m,
                n,
                nnz_a,
                csr_descr_a.inner,
                csr_val_a.as_ptr() as *const f32,
                csr_row_ptr_a.as_ptr(),
                csr_col_ind_a.as_ptr(),
                *(&percentage as *const T as *const f32),
                csr_descr_c.inner,
                csr_row_ptr_c.as_mut_ptr(),
                nnz_total,
                info.inner,
                temp_buffer,
            )
        };
        status_to_result(status)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let status = unsafe {
            rocsparse_dprune_csr2csr_nnz_by_percentage(
                handle.inner,
                m,
                n,
                nnz_a,
                csr_descr_a.inner,
                csr_val_a.as_ptr() as *const f64,
                csr_row_ptr_a.as_ptr(),
                csr_col_ind_a.as_ptr(),
                *(&percentage as *const T as *const f64),
                csr_descr_c.inner,
                csr_row_ptr_c.as_mut_ptr(),
                nnz_total,
                info.inner,
                temp_buffer,
            )
        };
        status_to_result(status)
    } else {
        Err(crate::rocsparse::error::Error::NotImplemented)
    }
}

/// Converts and prunes by percentage a sparse CSR matrix into a sparse CSR matrix
pub fn prune_csr2csr_by_percentage<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    nnz_a: i32,
    csr_descr_a: &MatrixDescriptor,
    csr_val_a: &[T],
    csr_row_ptr_a: &[i32],
    csr_col_ind_a: &[i32],
    percentage: T,
    csr_descr_c: &MatrixDescriptor,
    csr_val_c: &mut [T],
    csr_row_ptr_c: &[i32],
    csr_col_ind_c: &mut [i32],
    info: &MatrixInfo,
    temp_buffer: *mut c_void,
) -> Result<()>
where
    T: Copy + 'static,
{
    prune_csr2csr_by_percentage_typed(
        handle,
        m,
        n,
        nnz_a,
        csr_descr_a,
        csr_val_a,
        csr_row_ptr_a,
        csr_col_ind_a,
        percentage,
        csr_descr_c,
        csr_val_c,
        csr_row_ptr_c,
        csr_col_ind_c,
        info,
        temp_buffer,
    )
}

fn prune_csr2csr_by_percentage_typed<T: 'static>(
    handle: &Handle,
    m: i32,
    n: i32,
    nnz_a: i32,
    csr_descr_a: &MatrixDescriptor,
    csr_val_a: &[T],
    csr_row_ptr_a: &[i32],
    csr_col_ind_a: &[i32],
    percentage: T,
    csr_descr_c: &MatrixDescriptor,
    csr_val_c: &mut [T],
    csr_row_ptr_c: &[i32],
    csr_col_ind_c: &mut [i32],
    info: &MatrixInfo,
    temp_buffer: *mut c_void,
) -> Result<()> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let status = unsafe {
            rocsparse_sprune_csr2csr_by_percentage(
                handle.inner,
                m,
                n,
                nnz_a,
                csr_descr_a.inner,
                csr_val_a.as_ptr() as *const f32,
                csr_row_ptr_a.as_ptr(),
                csr_col_ind_a.as_ptr(),
                *(&percentage as *const T as *const f32),
                csr_descr_c.inner,
                csr_val_c.as_mut_ptr() as *mut f32,
                csr_row_ptr_c.as_ptr(),
                csr_col_ind_c.as_mut_ptr(),
                info.inner,
                temp_buffer,
            )
        };
        status_to_result(status)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let status = unsafe {
            rocsparse_dprune_csr2csr_by_percentage(
                handle.inner,
                m,
                n,
                nnz_a,
                csr_descr_a.inner,
                csr_val_a.as_ptr() as *const f64,
                csr_row_ptr_a.as_ptr(),
                csr_col_ind_a.as_ptr(),
                *(&percentage as *const T as *const f64),
                csr_descr_c.inner,
                csr_val_c.as_mut_ptr() as *mut f64,
                csr_row_ptr_c.as_ptr(),
                csr_col_ind_c.as_mut_ptr(),
                info.inner,
                temp_buffer,
            )
        };
        status_to_result(status)
    } else {
        Err(crate::rocsparse::error::Error::NotImplemented)
    }
}

/// Computes the buffer size required for dense to CSR conversion with pruning
pub fn prune_dense2csr_buffer_size<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    a: &[T],
    lda: i32,
    threshold: &T,
    descr: &MatrixDescriptor,
    csr_val: &[T],
    csr_row_ptr: &[i32],
    csr_col_ind: &[i32],
) -> Result<usize>
where
    T: Copy + 'static,
{
    let mut buffer_size = 0;
    let status = prune_dense2csr_buffer_size_typed(
        handle,
        m,
        n,
        a,
        lda,
        threshold,
        descr,
        csr_val,
        csr_row_ptr,
        csr_col_ind,
        &mut buffer_size,
    );
    status.map(|_| buffer_size)
}

fn prune_dense2csr_buffer_size_typed<T: 'static>(
    handle: &Handle,
    m: i32,
    n: i32,
    a: &[T],
    lda: i32,
    threshold: &T,
    descr: &MatrixDescriptor,
    csr_val: &[T],
    csr_row_ptr: &[i32],
    csr_col_ind: &[i32],
    buffer_size: &mut usize,
) -> Result<()> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let status = unsafe {
            rocsparse_sprune_dense2csr_buffer_size(
                handle.inner,
                m,
                n,
                a.as_ptr() as *const f32,
                lda,
                threshold as *const T as *const f32,
                descr.inner,
                csr_val.as_ptr() as *const f32,
                csr_row_ptr.as_ptr(),
                csr_col_ind.as_ptr(),
                buffer_size,
            )
        };
        status_to_result(status)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let status = unsafe {
            rocsparse_dprune_dense2csr_buffer_size(
                handle.inner,
                m,
                n,
                a.as_ptr() as *const f64,
                lda,
                threshold as *const T as *const f64,
                descr.inner,
                csr_val.as_ptr() as *const f64,
                csr_row_ptr.as_ptr(),
                csr_col_ind.as_ptr(),
                buffer_size,
            )
        };
        status_to_result(status)
    } else {
        Err(crate::rocsparse::error::Error::NotImplemented)
    }
}

/// Computes the number of non-zero elements per row and total non-zero elements
/// when converting dense matrix to CSR with pruning
pub fn prune_dense2csr_nnz<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    a: &[T],
    lda: i32,
    threshold: &T,
    descr: &MatrixDescriptor,
    csr_row_ptr: &mut [i32],
    temp_buffer: *mut c_void,
) -> Result<i32>
where
    T: Copy + 'static,
{
    let mut nnz_total = 0;
    let status = prune_dense2csr_nnz_typed(
        handle,
        m,
        n,
        a,
        lda,
        threshold,
        descr,
        csr_row_ptr,
        &mut nnz_total,
        temp_buffer,
    );
    status.map(|_| nnz_total)
}

fn prune_dense2csr_nnz_typed<T: 'static>(
    handle: &Handle,
    m: i32,
    n: i32,
    a: &[T],
    lda: i32,
    threshold: &T,
    descr: &MatrixDescriptor,
    csr_row_ptr: &mut [i32],
    nnz_total: &mut i32,
    temp_buffer: *mut c_void,
) -> Result<()> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let status = unsafe {
            rocsparse_sprune_dense2csr_nnz(
                handle.inner,
                m,
                n,
                a.as_ptr() as *const f32,
                lda,
                threshold as *const T as *const f32,
                descr.inner,
                csr_row_ptr.as_mut_ptr(),
                nnz_total,
                temp_buffer,
            )
        };
        status_to_result(status)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let status = unsafe {
            rocsparse_dprune_dense2csr_nnz(
                handle.inner,
                m,
                n,
                a.as_ptr() as *const f64,
                lda,
                threshold as *const T as *const f64,
                descr.inner,
                csr_row_ptr.as_mut_ptr(),
                nnz_total,
                temp_buffer,
            )
        };
        status_to_result(status)
    } else {
        Err(crate::rocsparse::error::Error::NotImplemented)
    }
}

/// Converts dense matrix to CSR format with pruning
pub fn prune_dense2csr<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    a: &[T],
    lda: i32,
    threshold: &T,
    descr: &MatrixDescriptor,
    csr_val: &mut [T],
    csr_row_ptr: &[i32],
    csr_col_ind: &mut [i32],
    temp_buffer: *mut c_void,
) -> Result<()>
where
    T: Copy + 'static,
{
    prune_dense2csr_typed(
        handle,
        m,
        n,
        a,
        lda,
        threshold,
        descr,
        csr_val,
        csr_row_ptr,
        csr_col_ind,
        temp_buffer,
    )
}

fn prune_dense2csr_typed<T: 'static>(
    handle: &Handle,
    m: i32,
    n: i32,
    a: &[T],
    lda: i32,
    threshold: &T,
    descr: &MatrixDescriptor,
    csr_val: &mut [T],
    csr_row_ptr: &[i32],
    csr_col_ind: &mut [i32],
    temp_buffer: *mut c_void,
) -> Result<()> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let status = unsafe {
            rocsparse_sprune_dense2csr(
                handle.inner,
                m,
                n,
                a.as_ptr() as *const f32,
                lda,
                threshold as *const T as *const f32,
                descr.inner,
                csr_val.as_mut_ptr() as *mut f32,
                csr_row_ptr.as_ptr(),
                csr_col_ind.as_mut_ptr(),
                temp_buffer,
            )
        };
        status_to_result(status)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let status = unsafe {
            rocsparse_dprune_dense2csr(
                handle.inner,
                m,
                n,
                a.as_ptr() as *const f64,
                lda,
                threshold as *const T as *const f64,
                descr.inner,
                csr_val.as_mut_ptr() as *mut f64,
                csr_row_ptr.as_ptr(),
                csr_col_ind.as_mut_ptr(),
                temp_buffer,
            )
        };
        status_to_result(status)
    } else {
        Err(crate::rocsparse::error::Error::NotImplemented)
    }
}

/// Computes the buffer size required for dense to CSR conversion with pruning by percentage
pub fn prune_dense2csr_by_percentage_buffer_size<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    a: &[T],
    lda: i32,
    percentage: T,
    descr: &MatrixDescriptor,
    csr_val: &[T],
    csr_row_ptr: &[i32],
    csr_col_ind: &[i32],
    info: &MatrixInfo,
) -> Result<usize>
where
    T: Copy + 'static,
{
    let mut buffer_size = 0;
    let status = prune_dense2csr_by_percentage_buffer_size_typed(
        handle,
        m,
        n,
        a,
        lda,
        percentage,
        descr,
        csr_val,
        csr_row_ptr,
        csr_col_ind,
        info,
        &mut buffer_size,
    );
    status.map(|_| buffer_size)
}

fn prune_dense2csr_by_percentage_buffer_size_typed<T: 'static>(
    handle: &Handle,
    m: i32,
    n: i32,
    a: &[T],
    lda: i32,
    percentage: T,
    descr: &MatrixDescriptor,
    csr_val: &[T],
    csr_row_ptr: &[i32],
    csr_col_ind: &[i32],
    info: &MatrixInfo,
    buffer_size: &mut usize,
) -> Result<()> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let status = unsafe {
            rocsparse_sprune_dense2csr_by_percentage_buffer_size(
                handle.inner,
                m,
                n,
                a.as_ptr() as *const f32,
                lda,
                *(&percentage as *const T as *const f32),
                descr.inner,
                csr_val.as_ptr() as *const f32,
                csr_row_ptr.as_ptr(),
                csr_col_ind.as_ptr(),
                info.inner,
                buffer_size,
            )
        };
        status_to_result(status)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let status = unsafe {
            rocsparse_dprune_dense2csr_by_percentage_buffer_size(
                handle.inner,
                m,
                n,
                a.as_ptr() as *const f64,
                lda,
                *(&percentage as *const T as *const f64),
                descr.inner,
                csr_val.as_ptr() as *const f64,
                csr_row_ptr.as_ptr(),
                csr_col_ind.as_ptr(),
                info.inner,
                buffer_size,
            )
        };
        status_to_result(status)
    } else {
        Err(crate::rocsparse::error::Error::NotImplemented)
    }
}

/// Computes the number of non-zero elements per row and total non-zero elements
/// when converting dense matrix to CSR with pruning by percentage
pub fn prune_dense2csr_nnz_by_percentage<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    a: &[T],
    lda: i32,
    percentage: T,
    descr: &MatrixDescriptor,
    csr_row_ptr: &mut [i32],
    info: &MatrixInfo,
    temp_buffer: *mut c_void,
) -> Result<i32>
where
    T: Copy + 'static,
{
    let mut nnz_total = 0;
    let status = prune_dense2csr_nnz_by_percentage_typed(
        handle,
        m,
        n,
        a,
        lda,
        percentage,
        descr,
        csr_row_ptr,
        &mut nnz_total,
        info,
        temp_buffer,
    );
    status.map(|_| nnz_total)
}

fn prune_dense2csr_nnz_by_percentage_typed<T: 'static>(
    handle: &Handle,
    m: i32,
    n: i32,
    a: &[T],
    lda: i32,
    percentage: T,
    descr: &MatrixDescriptor,
    csr_row_ptr: &mut [i32],
    nnz_total: &mut i32,
    info: &MatrixInfo,
    temp_buffer: *mut c_void,
) -> Result<()> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let status = unsafe {
            rocsparse_sprune_dense2csr_nnz_by_percentage(
                handle.inner,
                m,
                n,
                a.as_ptr() as *const f32,
                lda,
                *(&percentage as *const T as *const f32),
                descr.inner,
                csr_row_ptr.as_mut_ptr(),
                nnz_total,
                info.inner,
                temp_buffer,
            )
        };
        status_to_result(status)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let status = unsafe {
            rocsparse_dprune_dense2csr_nnz_by_percentage(
                handle.inner,
                m,
                n,
                a.as_ptr() as *const f64,
                lda,
                *(&percentage as *const T as *const f64),
                descr.inner,
                csr_row_ptr.as_mut_ptr(),
                nnz_total,
                info.inner,
                temp_buffer,
            )
        };
        status_to_result(status)
    } else {
        Err(crate::rocsparse::error::Error::NotImplemented)
    }
}

/// Converts dense matrix to CSR format with pruning by percentage
pub fn prune_dense2csr_by_percentage<T>(
    handle: &Handle,
    m: i32,
    n: i32,
    a: &[T],
    lda: i32,
    percentage: T,
    descr: &MatrixDescriptor,
    csr_val: &mut [T],
    csr_row_ptr: &[i32],
    csr_col_ind: &mut [i32],
    info: &MatrixInfo,
    temp_buffer: *mut c_void,
) -> Result<()>
where
    T: Copy + 'static,
{
    prune_dense2csr_by_percentage_typed(
        handle,
        m,
        n,
        a,
        lda,
        percentage,
        descr,
        csr_val,
        csr_row_ptr,
        csr_col_ind,
        info,
        temp_buffer,
    )
}

fn prune_dense2csr_by_percentage_typed<T: 'static>(
    handle: &Handle,
    m: i32,
    n: i32,
    a: &[T],
    lda: i32,
    percentage: T,
    descr: &MatrixDescriptor,
    csr_val: &mut [T],
    csr_row_ptr: &[i32],
    csr_col_ind: &mut [i32],
    info: &MatrixInfo,
    temp_buffer: *mut c_void,
) -> Result<()> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let status = unsafe {
            rocsparse_sprune_dense2csr_by_percentage(
                handle.inner,
                m,
                n,
                a.as_ptr() as *const f32,
                lda,
                *(&percentage as *const T as *const f32),
                descr.inner,
                csr_val.as_mut_ptr() as *mut f32,
                csr_row_ptr.as_ptr(),
                csr_col_ind.as_mut_ptr(),
                info.inner,
                temp_buffer,
            )
        };
        status_to_result(status)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let status = unsafe {
            rocsparse_dprune_dense2csr_by_percentage(
                handle.inner,
                m,
                n,
                a.as_ptr() as *const f64,
                lda,
                *(&percentage as *const T as *const f64),
                descr.inner,
                csr_val.as_mut_ptr() as *mut f64,
                csr_row_ptr.as_ptr(),
                csr_col_ind.as_mut_ptr(),
                info.inner,
                temp_buffer,
            )
        };
        status_to_result(status)
    } else {
        Err(crate::rocsparse::error::Error::NotImplemented)
    }
}