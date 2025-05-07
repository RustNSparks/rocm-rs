/*!
# Distributed Computing Support with Fields and Bricks

This module provides experimental support for distributed computation
using the Field and Brick abstractions in rocFFT.
*/

use crate::rocfft::bindings;
use crate::rocfft::error::{Error, Result, check_dimensions, check_error};
use std::marker::PhantomData;
use std::ptr;

/// Brick representing a portion of a data field for distributed computation
///
/// Bricks describe a specific portion of the data domain. Each brick defines:
/// - A position in the field's coordinate space
/// - Stride information for data access
/// - Which device the data resides on
///
/// # Note
///
/// This is an experimental feature in rocFFT.
pub struct Brick {
    handle: bindings::rocfft_brick,
    _marker: PhantomData<*mut ()>, // Mark as !Send and !Sync
}

impl Brick {
    /// Create a new brick for distributed computation
    ///
    /// # Arguments
    ///
    /// * `field_lower` - Array specifying the lower index (inclusive) in the field's coordinate space
    /// * `field_upper` - Array specifying the upper index (exclusive) in the field's coordinate space
    /// * `brick_stride` - Array specifying the brick's stride in memory
    /// * `device_id` - HIP device ID for the brick's data
    ///
    /// # Returns
    ///
    /// A result containing the newly created brick or an error
    ///
    /// # Example
    ///
    /// ```no_run
    ///
    ///
    /// // Create a brick describing a 32x32 segment at position (0,0,0)
    /// use rocm_rs::rocfft::Brick;
    /// let field_lower = vec![0, 0, 0];
    /// let field_upper = vec![32, 32, 1];  // Exclusive upper bound
    /// let brick_stride = vec![1, 32, 32*32]; // Row-major layout
    /// let device_id = 0; // First GPU
    ///
    /// let brick = Brick::new(&field_lower, &field_upper, &brick_stride, device_id)?;
    /// ```
    ///
    /// # Note
    ///
    /// This is an experimental feature in rocFFT.
    pub fn new(
        field_lower: &[usize],
        field_upper: &[usize],
        brick_stride: &[usize],
        device_id: i32,
    ) -> Result<Self> {
        // Validate parameters
        if field_lower.len() != field_upper.len() || field_lower.len() != brick_stride.len() {
            return Err(Error::InvalidDimensions);
        }

        // Make sure all dimensions are valid
        for (i, (&lower, &upper)) in field_lower.iter().zip(field_upper.iter()).enumerate() {
            if lower >= upper {
                return Err(Error::InvalidDimensions);
            }
        }

        let dim = field_lower.len();
        let mut handle: bindings::rocfft_brick = ptr::null_mut();

        unsafe {
            check_error(bindings::rocfft_brick_create(
                &mut handle,
                field_lower.as_ptr(),
                field_upper.as_ptr(),
                brick_stride.as_ptr(),
                dim,
                device_id,
            ))?;
        }

        Ok(Brick {
            handle,
            _marker: PhantomData,
        })
    }

    /// Get the internal handle (for use in other rocFFT functions)
    pub(crate) fn as_ptr(&self) -> bindings::rocfft_brick {
        self.handle
    }
}

impl Drop for Brick {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                bindings::rocfft_brick_destroy(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

/// A field describing data decomposition for distributed computation
///
/// Fields are collections of Bricks that together describe how data is distributed
/// across multiple devices. Fields are used to specify input and output data
/// decompositions for transforms.
///
/// # Note
///
/// This is an experimental feature in rocFFT.
pub struct Field {
    handle: bindings::rocfft_field,
    _marker: PhantomData<*mut ()>, // Mark as !Send and !Sync
}

impl Field {
    /// Create a new field for distributed data decomposition
    ///
    /// # Returns
    ///
    /// A result containing the newly created field or an error
    ///
    /// # Example
    ///
    /// ```no_run
    ///
    ///
    /// // Create a field to hold the decomposition
    /// use rocm_rs::rocfft::Field;
    /// let mut field = Field::new()?;
    ///
    /// // Create bricks and add them to the field
    /// // (example assumes you've already created brick1, brick2, etc.)
    /// // field.add_brick(&brick1)?;
    /// // field.add_brick(&brick2)?;
    /// ```
    ///
    /// # Note
    ///
    /// This is an experimental feature in rocFFT.
    pub fn new() -> Result<Self> {
        let mut handle: bindings::rocfft_field = ptr::null_mut();

        unsafe {
            check_error(bindings::rocfft_field_create(&mut handle))?;
        }

        Ok(Field {
            handle,
            _marker: PhantomData,
        })
    }

    /// Add a brick to the field
    ///
    /// # Arguments
    ///
    /// * `brick` - The brick to add to the field
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Important Note
    ///
    /// The order in which bricks are added to the field is significant and corresponds
    /// to the order of buffer pointers provided to `Plan::execute`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocfft::{Brick, Field};
    ///
    ///
    /// let mut field = Field::new()?;
    ///
    /// // Create brick for the first part of the domain
    /// let brick1 = Brick::new(&[0, 0], &[64, 64], &[1, 64], 0)?;
    ///
    /// // Create brick for the second part of the domain
    /// let brick2 = Brick::new(&[64, 0], &[128, 64], &[1, 64], 1)?;
    ///
    /// // Add bricks to the field - order matters!
    /// field.add_brick(&brick1)?;
    /// field.add_brick(&brick2)?;
    /// ```
    pub fn add_brick(&mut self, brick: &Brick) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        unsafe {
            check_error(bindings::rocfft_field_add_brick(
                self.handle,
                brick.as_ptr(),
            ))
        }
    }

    /// Get the internal handle (for use in other rocFFT functions)
    pub(crate) fn as_ptr(&self) -> bindings::rocfft_field {
        self.handle
    }
}

impl Drop for Field {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                bindings::rocfft_field_destroy(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

// These objects are not safe to send between threads because they contain
// raw pointers and device-specific state
