/*!
# FFT Plan Management

This module provides the core Plan type for defining FFT transforms.
*/

use std::ptr;
use std::marker::PhantomData;
use crate::rocfft::error::{Error, Result, check_error, check_dimensions};
use crate::rocfft::bindings;
use crate::rocfft::description::PlanDescription;
use crate::rocfft::execution::ExecutionInfo;

/// The type of transform to be performed
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TransformType {
    /// Complex forward FFT (typically uses e^(-j*2*pi*n/N))
    ComplexForward,
    /// Complex inverse FFT (typically uses e^(j*2*pi*n/N))
    ComplexInverse,
    /// Real forward FFT (real input, complex output)
    RealForward,
    /// Real inverse FFT (complex input, real output)
    RealInverse,
}

impl From<TransformType> for u32 {
    fn from(transform_type: TransformType) -> Self {
        match transform_type {
            TransformType::ComplexForward => bindings::rocfft_transform_type_e_rocfft_transform_type_complex_forward,
            TransformType::ComplexInverse => bindings::rocfft_transform_type_e_rocfft_transform_type_complex_inverse,
            TransformType::RealForward => bindings::rocfft_transform_type_e_rocfft_transform_type_real_forward,
            TransformType::RealInverse => bindings::rocfft_transform_type_e_rocfft_transform_type_real_inverse,
        }
    }
}

/// The numerical precision to be used
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Precision {
    /// Single precision (32-bit floating point)
    Single,
    /// Double precision (64-bit floating point)
    Double,
    /// Half precision (16-bit floating point)
    Half,
}

impl From<Precision> for u32 {
    fn from(precision: Precision) -> Self {
        match precision {
            Precision::Single => bindings::rocfft_precision_e_rocfft_precision_single,
            Precision::Double => bindings::rocfft_precision_e_rocfft_precision_double,
            Precision::Half => bindings::rocfft_precision_e_rocfft_precision_half,
        }
    }
}

/// Specifies whether the transform is in-place or out-of-place
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PlacementType {
    /// Input and output buffers are the same (in-place transform)
    InPlace,
    /// Input and output buffers are different (out-of-place transform)
    NotInPlace,
}

impl From<PlacementType> for u32 {
    fn from(placement: PlacementType) -> Self {
        match placement {
            PlacementType::InPlace => bindings::rocfft_result_placement_e_rocfft_placement_inplace,
            PlacementType::NotInPlace => bindings::rocfft_result_placement_e_rocfft_placement_notinplace,
        }
    }
}

/// The type and format of data arrays
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ArrayType {
    /// Complex data stored in interleaved format (real and imaginary parts adjacent in memory)
    ComplexInterleaved,
    /// Complex data stored in planar format (real and imaginary parts in separate arrays)
    ComplexPlanar,
    /// Real data (no imaginary component)
    Real,
    /// Hermitian data in interleaved format (for real transforms)
    HermitianInterleaved,
    /// Hermitian data in planar format (for real transforms)
    HermitianPlanar,
    /// Array type is not set
    Unset,
}

impl From<ArrayType> for u32 {
    fn from(array_type: ArrayType) -> Self {
        match array_type {
            ArrayType::ComplexInterleaved => bindings::rocfft_array_type_e_rocfft_array_type_complex_interleaved,
            ArrayType::ComplexPlanar => bindings::rocfft_array_type_e_rocfft_array_type_complex_planar,
            ArrayType::Real => bindings::rocfft_array_type_e_rocfft_array_type_real,
            ArrayType::HermitianInterleaved => bindings::rocfft_array_type_e_rocfft_array_type_hermitian_interleaved,
            ArrayType::HermitianPlanar => bindings::rocfft_array_type_e_rocfft_array_type_hermitian_planar,
            ArrayType::Unset => bindings::rocfft_array_type_e_rocfft_array_type_unset,
        }
    }
}

/// An FFT plan that defines all parameters of a transform
pub struct Plan {
    handle: bindings::rocfft_plan,
    _marker: PhantomData<*mut ()>, // Mark as !Send and !Sync
}

impl Plan {
    /// Create a new FFT plan with the given parameters
    ///
    /// # Arguments
    ///
    /// * `placement` - Whether the transform is in-place or out-of-place
    /// * `transform_type` - The type of transform to perform
    /// * `precision` - The numerical precision to use
    /// * `dimensions` - The number of dimensions (1, 2, or 3)
    /// * `lengths` - The size of the data in each dimension (length must match `dimensions`)
    /// * `number_of_transforms` - Number of transforms of the same size to perform (batch size)
    /// * `description` - Optional plan description for additional parameters
    ///
    /// # Returns
    ///
    /// A result containing the newly created plan or an error
    ///
    /// # Example
    ///
    /// ```no_run
    ///
    /// // Create a plan for a 1D complex forward FFT of length 1024
    /// use rocm_rs::rocfft::{PlacementType, Plan, Precision, TransformType};
    /// let lengths = vec![1024];
    /// let plan = Plan::new(
    ///     PlacementType::InPlace,
    ///     TransformType::ComplexForward,
    ///     Precision::Single,
    ///     1,
    ///     &lengths,
    ///     1,
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn new(
        placement: PlacementType,
        transform_type: TransformType,
        precision: Precision,
        dimensions: usize,
        lengths: &[usize],
        number_of_transforms: usize,
        description: Option<&PlanDescription>,
    ) -> Result<Self> {
        // Validate dimensions
        check_dimensions(dimensions)?;

        if lengths.len() != dimensions {
            return Err(Error::InvalidDimensions);
        }

        let mut handle: bindings::rocfft_plan = ptr::null_mut();

        unsafe {
            let desc_ptr = match description {
                Some(desc) => desc.as_ptr(),
                None => ptr::null_mut(),
            };

            check_error(bindings::rocfft_plan_create(
                &mut handle,
                placement.into(),
                transform_type.into(),
                precision.into(),
                dimensions,
                lengths.as_ptr(),
                number_of_transforms,
                desc_ptr,
            ))?;
        }

        Ok(Plan {
            handle,
            _marker: PhantomData,
        })
    }

    /// Execute the plan with the given input and output buffers
    ///
    /// # Arguments
    ///
    /// * `input` - Array of input buffer pointers (usually just one pointer for interleaved formats,
    ///             two pointers for planar formats, or one per brick if using fields)
    /// * `output` - Array of output buffer pointers (can be empty for in-place transforms)
    /// * `info` - Optional execution info for setting work buffers or streams
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Safety
    ///
    /// This function is marked as safe, but it requires that the input and output
    /// buffer pointers point to valid GPU memory of sufficient size for the transform.
    /// It's the caller's responsibility to ensure this.
    pub fn execute(
        &mut self,
        input: &[*mut std::ffi::c_void],
        output: &[*mut std::ffi::c_void],
        info: Option<&mut ExecutionInfo>,
    ) -> Result<()> {
        if input.is_empty() {
            return Err(Error::InvalidArgValue);
        }

        let in_ptr_array = input.as_ptr() as *mut *mut std::ffi::c_void;

        let out_ptr_array = if output.is_empty() {
            ptr::null_mut()
        } else {
            output.as_ptr() as *mut *mut std::ffi::c_void
        };

        let info_ptr = match info {
            Some(exec_info) => exec_info.as_ptr(),
            None => ptr::null_mut(),
        };

        unsafe {
            check_error(bindings::rocfft_execute(
                self.handle,
                in_ptr_array,
                out_ptr_array,
                info_ptr,
            ))
        }
    }

    /// Get the work buffer size required for this plan
    ///
    /// # Returns
    ///
    /// A result containing the work buffer size in bytes
    pub fn get_work_buffer_size(&self) -> Result<usize> {
        let mut size: usize = 0;

        unsafe {
            check_error(bindings::rocfft_plan_get_work_buffer_size(
                self.handle,
                &mut size,
            ))?;
        }

        Ok(size)
    }

    /// Print detailed information about this plan to stdout (for debugging)
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    pub fn print_info(&self) -> Result<()> {
        unsafe {
            check_error(bindings::rocfft_plan_get_print(self.handle))
        }
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                bindings::rocfft_plan_destroy(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}