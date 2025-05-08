//! Sparse vector types

use std::mem::MaybeUninit;
use std::marker::PhantomData;
use std::ffi::c_void;
use crate::rocsparse::error::{Result, status_to_result};
use crate::rocsparse::{rocsparse_create_spvec_descr, rocsparse_datatype, rocsparse_destroy_spvec_descr, rocsparse_indextype, rocsparse_spvec_descr};
use crate::rocsparse::descriptor::IndexBase;

/// Sparse vectors
pub struct SparseVector<T> {
    pub(crate) inner: rocsparse_spvec_descr,
    _phantom: PhantomData<T>,
}

impl<T> SparseVector<T> {
    /// Create a new sparse vector
    pub unsafe fn new(
        size: i64,
        nnz: i64,
        indices: *mut c_void,
        values: *mut c_void,
        idx_type: rocsparse_indextype,
        idx_base: IndexBase,
        data_type: rocsparse_datatype,
    ) -> Result<Self> {
        let mut descr = MaybeUninit::uninit();
        let status = unsafe {
            rocsparse_create_spvec_descr(
                descr.as_mut_ptr(),
                size,
                nnz,
                indices,
                values,
                idx_type,
                idx_base.into(),
                data_type,
            )
        };
        status_to_result(status)?;
        let descr = unsafe { descr.assume_init() };
        Ok(Self {
            inner: descr,
            _phantom: PhantomData,
        })
    }
}

impl<T> Drop for SparseVector<T> {
    fn drop(&mut self) {
        unsafe {
            // Ignore error on drop
            let _ = rocsparse_destroy_spvec_descr(self.inner);
        }
    }
}