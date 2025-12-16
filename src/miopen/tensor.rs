// src/miopen/tensor.rs

use crate::miopen::error::{Error, Result};
use crate::miopen::ffi;
use crate::miopen::handle::Handle;
use std::ptr;

// pub type DataType = ffi::miopenDataType_t;

/// MIOpen data types
#[repr(u32)]
pub enum DataType {
    MiopenHalf = ffi::miopenDataType_t_miopenHalf,
    MiopenFloat = ffi::miopenDataType_t_miopenFloat,
    MiopenInt32 = ffi::miopenDataType_t_miopenInt32,
    MiopenInt8 = ffi::miopenDataType_t_miopenInt8,
    MiopenBFloat16 = ffi::miopenDataType_t_miopenBFloat16,
    MiopenDouble = ffi::miopenDataType_t_miopenDouble,
    MiopenInt64 = ffi::miopenDataType_t_miopenInt64,
}

impl TryFrom<u32> for DataType {
    type Error = Error;

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        match value {
            ffi::miopenDataType_t_miopenHalf => Ok(DataType::MiopenHalf),
            ffi::miopenDataType_t_miopenFloat => Ok(DataType::MiopenFloat),
            ffi::miopenDataType_t_miopenInt32 => Ok(DataType::MiopenInt32),
            ffi::miopenDataType_t_miopenInt8 => Ok(DataType::MiopenInt8),
            ffi::miopenDataType_t_miopenBFloat16 => Ok(DataType::MiopenBFloat16),
            ffi::miopenDataType_t_miopenDouble => Ok(DataType::MiopenDouble),
            ffi::miopenDataType_t_miopenInt64 => Ok(DataType::MiopenInt64),
            _ => Err(Error::new(ffi::miopenStatus_t_miopenStatusUnknownError)),
        }
    }
}

/// MIOpen tensor layout
pub type TensorLayout = ffi::miopenTensorLayout_t;

/// Safe wrapper for MIOpen tensor descriptor
pub struct TensorDescriptor {
    desc: ffi::miopenTensorDescriptor_t,
}

impl TensorDescriptor {
    /// Create a new tensor descriptor
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe { ffi::miopenCreateTensorDescriptor(&mut desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Set the descriptor for a 4D tensor (NCHW format)
    pub fn set_4d(&mut self, data_type: DataType, n: i32, c: i32, h: i32, w: i32) -> Result<()> {
        let status =
            unsafe { ffi::miopenSet4dTensorDescriptor(self.desc, data_type as u32, n, c, h, w) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Create a 4D tensor (NCHW format)
    pub fn new_4d(data_type: DataType, n: i32, c: i32, h: i32, w: i32) -> Result<Self> {
        let mut desc = Self::new()?;
        desc.set_4d(data_type, n, c, h, w)?;
        Ok(desc)
    }

    /// Set the descriptor for a 4D tensor with strides
    pub fn set_4d_ex(
        &mut self,
        data_type: DataType,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        n_stride: i32,
        c_stride: i32,
        h_stride: i32,
        w_stride: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSet4dTensorDescriptorEx(
                self.desc,
                data_type as u32,
                n,
                c,
                h,
                w,
                n_stride,
                c_stride,
                h_stride,
                w_stride,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Set the descriptor for an N-dimensional tensor with specific layout
    pub fn set_nd_with_layout(
        &mut self,
        data_type: DataType,
        layout: TensorLayout,
        dims: &[i32],
    ) -> Result<()> {
        let num_dims = dims.len() as i32;

        let status = unsafe {
            ffi::miopenSetNdTensorDescriptorWithLayout(
                self.desc,
                data_type as u32,
                layout,
                dims.as_ptr(),
                num_dims,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the details of a 4D tensor descriptor
    pub fn get_4d(&self) -> Result<(DataType, i32, i32, i32, i32, i32, i32, i32, i32)> {
        let mut data_type = 0;
        let mut n = 0;
        let mut c = 0;
        let mut h = 0;
        let mut w = 0;
        let mut n_stride = 0;
        let mut c_stride = 0;
        let mut h_stride = 0;
        let mut w_stride = 0;

        let status = unsafe {
            ffi::miopenGet4dTensorDescriptor(
                self.desc,
                &mut data_type,
                &mut n,
                &mut c,
                &mut h,
                &mut w,
                &mut n_stride,
                &mut c_stride,
                &mut h_stride,
                &mut w_stride,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((
            DataType::try_from(data_type)?,
            n,
            c,
            h,
            w,
            n_stride,
            c_stride,
            h_stride,
            w_stride,
        ))
    }

    /// Set the descriptor for an N-dimensional tensor
    pub fn set_nd(&mut self, data_type: DataType, dims: &[i32], strides: &[i32]) -> Result<()> {
        let nb_dims = dims.len() as i32;

        if nb_dims != strides.len() as i32 {
            return Err(Error::new(ffi::miopenStatus_t_miopenStatusBadParm));
        }

        let status = unsafe {
            ffi::miopenSetTensorDescriptor(
                self.desc,
                data_type as u32,
                nb_dims,
                dims.as_ptr(),
                strides.as_ptr(),
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Gets the size of tensor dimensions
    pub fn get_size(&self) -> Result<i32> {
        let mut size = 0;

        let status = unsafe { ffi::miopenGetTensorDescriptorSize(self.desc, &mut size) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(size)
    }

    /// Get the descriptor details for an N-dimensional tensor
    pub fn get_nd(
        &self,
        dims_capacity: usize,
        strides_capacity: usize,
    ) -> Result<(DataType, Vec<i32>, Vec<i32>)> {
        let mut data_type = 0;
        let mut dims = vec![0; dims_capacity];
        let mut strides = vec![0; strides_capacity];

        let status = unsafe {
            ffi::miopenGetTensorDescriptor(
                self.desc,
                &mut data_type,
                dims.as_mut_ptr(),
                strides.as_mut_ptr(),
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((DataType::try_from(data_type)?, dims, strides))
    }

    /// Get the number of bytes for a tensor
    pub fn get_num_bytes(&self) -> Result<usize> {
        let mut num_bytes = 0;

        let status = unsafe { ffi::miopenGetTensorNumBytes(self.desc, &mut num_bytes) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(num_bytes)
    }

    /// Transform tensor from one layout to another
    pub unsafe fn transform(
        &self,
        handle: &Handle,
        alpha: &[u8],
        x_desc: &TensorDescriptor,
        x: *const ::std::os::raw::c_void,
        beta: &[u8],
        y: *mut ::std::os::raw::c_void,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenTransformTensor(
                handle.as_raw(),
                alpha.as_ptr() as *const ::std::os::raw::c_void,
                x_desc.as_raw(),
                x,
                beta.as_ptr() as *const ::std::os::raw::c_void,
                self.as_raw(),
                y,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Set tensor to a single value
    pub unsafe fn set_tensor(
        &self,
        handle: &Handle,
        y: *mut ::std::os::raw::c_void,
        alpha: &[u8],
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetTensor(
                handle.as_raw(),
                self.as_raw(),
                y,
                alpha.as_ptr() as *const ::std::os::raw::c_void,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Scale a tensor by a single value
    pub unsafe fn scale_tensor(
        &self,
        handle: &Handle,
        y: *mut ::std::os::raw::c_void,
        alpha: &[u8],
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenScaleTensor(
                handle.as_raw(),
                self.as_raw(),
                y,
                alpha.as_ptr() as *const ::std::os::raw::c_void,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Execute element-wise tensor operations
    pub unsafe fn op_tensor(
        &self,
        handle: &Handle,
        tensor_op: ffi::miopenTensorOp_t,
        alpha1: &[u8],
        a_desc: &TensorDescriptor,
        a: *const ::std::os::raw::c_void,
        alpha2: &[u8],
        b_desc: &TensorDescriptor,
        b: *const ::std::os::raw::c_void,
        beta: &[u8],
        c: *mut ::std::os::raw::c_void,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenOpTensor(
                handle.as_raw(),
                tensor_op,
                alpha1.as_ptr() as *const ::std::os::raw::c_void,
                a_desc.as_raw(),
                a,
                alpha2.as_ptr() as *const ::std::os::raw::c_void,
                b_desc.as_raw(),
                b,
                beta.as_ptr() as *const ::std::os::raw::c_void,
                self.as_raw(),
                c,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the raw descriptor handle
    pub fn as_raw(&self) -> ffi::miopenTensorDescriptor_t {
        self.desc
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe {
                let _ = ffi::miopenDestroyTensorDescriptor(self.desc);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.desc = ptr::null_mut();
        }
    }
}

/// Safe wrapper for MIOpen sequence tensor descriptor
pub struct SeqTensorDescriptor {
    desc: ffi::miopenSeqTensorDescriptor_t,
}

impl SeqTensorDescriptor {
    /// Create a new sequence tensor descriptor
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe { ffi::miopenCreateSeqTensorDescriptor(&mut desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Set the descriptor for a RNN sequence data tensor
    pub fn set_rnn_data_seq_tensor(
        &mut self,
        data_type: DataType,
        layout: ffi::miopenRNNBaseLayout_t,
        max_sequence_len: i32,
        batch_size: i32,
        vector_size: i32,
        sequence_len_array: &[i32],
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetRNNDataSeqTensorDescriptor(
                self.desc,
                data_type as u32,
                layout,
                max_sequence_len,
                batch_size,
                vector_size,
                sequence_len_array.as_ptr(),
                ptr::null_mut(), // paddingMarker, should be NULL
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the descriptor details for a RNN sequence data tensor
    pub fn get_rnn_data_seq_tensor(
        &self,
        sequence_len_array_limit: i32,
    ) -> Result<(
        DataType,
        ffi::miopenRNNBaseLayout_t,
        i32,
        i32,
        i32,
        Vec<i32>,
    )> {
        let mut data_type = 0;
        let mut layout = 0;
        let mut max_sequence_len = 0;
        let mut batch_size = 0;
        let mut vector_size = 0;
        let mut sequence_len_array = vec![0; sequence_len_array_limit as usize];

        let status = unsafe {
            ffi::miopenGetRNNDataSeqTensorDescriptor(
                self.desc,
                &mut data_type,
                &mut layout,
                &mut max_sequence_len,
                &mut batch_size,
                &mut vector_size,
                sequence_len_array_limit,
                if sequence_len_array_limit > 0 {
                    sequence_len_array.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                ptr::null_mut(), // paddingMarker, should be NULL
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((
            DataType::try_from(data_type)?,
            layout,
            max_sequence_len,
            batch_size,
            vector_size,
            sequence_len_array,
        ))
    }

    /// Get the raw descriptor handle
    pub fn as_raw(&self) -> ffi::miopenSeqTensorDescriptor_t {
        self.desc
    }
}

impl Drop for SeqTensorDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe {
                let _ = ffi::miopenDestroySeqTensorDescriptor(self.desc);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.desc = ptr::null_mut();
        }
    }
}
