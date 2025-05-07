/*!
# Advanced Plan Configuration

This module provides the PlanDescription type for configuring advanced aspects
of FFT plans such as data layout, communication, and scaling factors.
*/

use crate::rocfft::bindings;
use crate::rocfft::error::{Error, Result, check_error};
use crate::rocfft::field::Field;
use crate::rocfft::plan::ArrayType;
use std::marker::PhantomData;
use std::ptr;

/// Communication library type for distributed transforms
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CommType {
    /// No communication library (single-node operation)
    None,
    /// MPI communication library
    MPI,
}

impl From<CommType> for u32 {
    fn from(comm_type: CommType) -> Self {
        match comm_type {
            CommType::None => bindings::rocfft_comm_type_e_rocfft_comm_none,
            CommType::MPI => bindings::rocfft_comm_type_e_rocfft_comm_mpi,
        }
    }
}

/// A description of advanced plan parameters
///
/// This object allows setting additional details about a transform, such as:
/// - Data layout (strides, offsets, distances)
/// - Array formats (interleaved vs planar)
/// - Communication for distributed transforms
/// - Scale factors
/// - Fields for distributed data
pub struct PlanDescription {
    handle: bindings::rocfft_plan_description,
    _marker: PhantomData<*mut ()>, // Mark as !Send and !Sync
}

impl PlanDescription {
    /// Create a new plan description
    ///
    /// # Returns
    ///
    /// A result containing the newly created plan description or an error
    ///
    /// # Example
    ///
    /// ```no_run
    ///
    /// use rocm_rs::rocfft::description::PlanDescription;
    ///
    /// let description = PlanDescription::new()?;
    /// ```
    pub fn new() -> Result<Self> {
        let mut handle: bindings::rocfft_plan_description = ptr::null_mut();

        unsafe {
            check_error(bindings::rocfft_plan_description_create(&mut handle))?;
        }

        Ok(PlanDescription {
            handle,
            _marker: PhantomData,
        })
    }

    /// Set the scaling factor for the transform
    ///
    /// # Arguments
    ///
    /// * `scale_factor` - Factor to multiply each transform element by
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Example
    ///
    /// ```no_run
    ///
    /// use rocm_rs::rocfft::description::PlanDescription;
    /// let mut description = PlanDescription::new()?;
    ///
    /// // Scale by 1/N for an inverse transform
    /// description.set_scale_factor(1.0 / 1024.0)?;
    /// ```
    pub fn set_scale_factor(&mut self, scale_factor: f64) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        if !scale_factor.is_finite() {
            return Err(Error::InvalidArgValue);
        }

        unsafe {
            check_error(bindings::rocfft_plan_description_set_scale_factor(
                self.handle,
                scale_factor,
            ))
        }
    }

    /// Set the data layout for input and output buffers
    ///
    /// # Arguments
    ///
    /// * `in_array_type` - Array type of input buffer
    /// * `out_array_type` - Array type of output buffer
    /// * `in_offsets` - Offsets to start of data in input buffer (in element units)
    /// * `out_offsets` - Offsets to start of data in output buffer (in element units)
    /// * `in_strides` - Strides in each dimension of input buffer
    /// * `in_distance` - Distance between start of each data instance in input buffer
    /// * `out_strides` - Strides in each dimension of output buffer
    /// * `out_distance` - Distance between start of each data instance in output buffer
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Example
    ///
    /// ```no_run
    ///
    /// use rocm_rs::rocfft::description::PlanDescription;
    /// use rocm_rs::rocfft::plan::ArrayType;
    /// let mut desc = PlanDescription::new()?;
    ///
    /// // Set custom strides for a 2D transform with non-contiguous memory layout
    /// desc.set_data_layout(
    ///     ArrayType::ComplexInterleaved,  // Input type
    ///     ArrayType::ComplexInterleaved,  // Output type
    ///     Some(&[0]),                     // Input offset (start at element 0)
    ///     Some(&[0]),                     // Output offset (start at element 0)
    ///     Some(&[2, 2048]),               // Input strides (2 elements between adjacent elements in dim 0,
    ///                                     // 2048 elements between adjacent elements in dim 1)
    ///     2048 * 1024,                    // Input batch distance
    ///     Some(&[2, 2048]),               // Output strides (same as input in this case)
    ///     2048 * 1024,                    // Output batch distance
    /// )?;
    /// ```
    pub fn set_data_layout(
        &mut self,
        in_array_type: ArrayType,
        out_array_type: ArrayType,
        in_offsets: Option<&[usize]>,
        out_offsets: Option<&[usize]>,
        in_strides: Option<&[usize]>,
        in_distance: usize,
        out_strides: Option<&[usize]>,
        out_distance: usize,
    ) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        let in_offsets_ptr = match in_offsets {
            Some(offsets) => offsets.as_ptr(),
            None => ptr::null(),
        };

        let out_offsets_ptr = match out_offsets {
            Some(offsets) => offsets.as_ptr(),
            None => ptr::null(),
        };

        let (in_strides_ptr, in_strides_size) = match in_strides {
            Some(strides) => (strides.as_ptr(), strides.len()),
            None => (ptr::null(), 0),
        };

        let (out_strides_ptr, out_strides_size) = match out_strides {
            Some(strides) => (strides.as_ptr(), strides.len()),
            None => (ptr::null(), 0),
        };

        unsafe {
            check_error(bindings::rocfft_plan_description_set_data_layout(
                self.handle,
                in_array_type.into(),
                out_array_type.into(),
                in_offsets_ptr,
                out_offsets_ptr,
                in_strides_size,
                in_strides_ptr,
                in_distance,
                out_strides_size,
                out_strides_ptr,
                out_distance,
            ))
        }
    }

    /// Set the communication library for distributed transforms
    ///
    /// # Arguments
    ///
    /// * `comm_type` - Type of communication library to use
    /// * `comm_handle` - Handle to communication-library-specific state (e.g., MPI_Comm)
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocfft::description::{CommType, PlanDescription};
    ///
    /// let mut desc = PlanDescription::new()?;
    ///
    /// // Set up for MPI distributed transforms
    /// let mpi_comm = /* get MPI communicator handle */;
    /// unsafe { desc.set_comm(CommType::MPI, mpi_comm as *mut std::ffi::c_void)?; }
    /// ```
    pub unsafe fn set_comm(
        &mut self,
        comm_type: CommType,
        comm_handle: *mut std::ffi::c_void,
    ) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        unsafe {
            check_error(bindings::rocfft_plan_description_set_comm(
                self.handle,
                comm_type.into(),
                comm_handle,
            ))
        }
    }

    /// Add a field to the plan description for input data decomposition
    ///
    /// # Arguments
    ///
    /// * `field` - The field defining the data decomposition
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Example
    ///
    /// ```no_run
    ///
    ///
    ///
    /// use rocm_rs::rocfft::description::PlanDescription;
    /// use rocm_rs::rocfft::field::Field;
    /// let mut desc = PlanDescription::new()?;
    /// let mut field = Field::new()?;
    ///
    /// // Add bricks to the field
    /// // field.add_brick(&brick1)?;
    /// // field.add_brick(&brick2)?;
    ///
    /// // Set field as input
    /// desc.add_infield(&field)?;
    /// ```
    ///
    /// # Note
    ///
    /// This is an experimental feature.
    pub fn add_infield(&mut self, field: &Field) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        unsafe {
            check_error(bindings::rocfft_plan_description_add_infield(
                self.handle,
                field.as_ptr(),
            ))
        }
    }

    /// Add a field to the plan description for output data decomposition
    ///
    /// # Arguments
    ///
    /// * `field` - The field defining the data decomposition
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Example
    ///
    /// ```no_run
    ///
    ///
    /// use rocm_rs::rocfft::description::PlanDescription;
    /// use rocm_rs::rocfft::field::Field;
    /// let mut desc = PlanDescription::new()?;
    /// let mut field = Field::new()?;
    ///
    /// // Add bricks to the field
    /// // field.add_brick(&brick1)?;
    /// // field.add_brick(&brick2)?;
    ///
    /// // Set field as output
    /// desc.add_outfield(&field)?;
    /// ```
    ///
    /// # Note
    ///
    /// This is an experimental feature.
    pub fn add_outfield(&mut self, field: &Field) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        unsafe {
            check_error(bindings::rocfft_plan_description_add_outfield(
                self.handle,
                field.as_ptr(),
            ))
        }
    }

    /// Get the internal handle (for use in other rocFFT functions)
    pub(crate) fn as_ptr(&self) -> bindings::rocfft_plan_description {
        self.handle
    }
}

impl Drop for PlanDescription {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                bindings::rocfft_plan_description_destroy(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

// Prevent sending a plan description between threads as it's not guaranteed to be thread-safe
