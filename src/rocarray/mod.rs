// src/rocarray/mod.rs - Fixed ROCArray with multidimensional support

use crate::error::Result;
use crate::hip::memory::PendingCopy;
use crate::hip::{DeviceMemory, Stream};
use std::fmt;
use std::marker::PhantomData;

pub mod kernels;
pub mod random;
pub mod sorting;

/// Shape information for multidimensional arrays
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
    strides: Vec<usize>,
}

impl Shape {
    /// Create a new shape from dimensions
    pub fn new(dims: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&dims);
        Self { dims, strides }
    }

    /// Create a 1D shape
    pub fn new_1d(len: usize) -> Self {
        Self::new(vec![len])
    }

    /// Create a 2D shape
    pub fn new_2d(rows: usize, cols: usize) -> Self {
        Self::new(vec![rows, cols])
    }

    /// Create a 3D shape
    pub fn new_3d(depth: usize, rows: usize, cols: usize) -> Self {
        Self::new(vec![depth, rows, cols])
    }

    /// Get dimensions
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get strides
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    /// Check if shapes are compatible for broadcasting
    pub fn can_broadcast_with(&self, other: &Shape) -> bool {
        let max_ndim = self.ndim().max(other.ndim());

        for i in 0..max_ndim {
            let dim1 = self
                .dims
                .get(self.ndim().saturating_sub(i + 1))
                .copied()
                .unwrap_or(1);
            let dim2 = other
                .dims
                .get(other.ndim().saturating_sub(i + 1))
                .copied()
                .unwrap_or(1);

            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false;
            }
        }
        true
    }

    /// Broadcast two shapes together
    pub fn broadcast_with(&self, other: &Shape) -> Option<Shape> {
        if !self.can_broadcast_with(other) {
            return None;
        }

        let max_ndim = self.ndim().max(other.ndim());
        let mut result_dims = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let dim1 = self
                .dims
                .get(self.ndim().saturating_sub(i + 1))
                .copied()
                .unwrap_or(1);
            let dim2 = other
                .dims
                .get(other.ndim().saturating_sub(i + 1))
                .copied()
                .unwrap_or(1);
            result_dims.push(dim1.max(dim2));
        }

        result_dims.reverse();
        Some(Shape::new(result_dims))
    }

    /// Convert flat index to multidimensional indices
    pub fn unravel_index(&self, index: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.ndim());
        let mut remaining = index;

        for &stride in &self.strides {
            indices.push(remaining / stride);
            remaining %= stride;
        }

        indices
    }

    /// Convert multidimensional indices to flat index
    pub fn ravel_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.ndim() {
            return None;
        }

        let mut flat_index = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.dims[i] {
                return None;
            }
            flat_index += idx * self.strides[i];
        }

        Some(flat_index)
    }

    /// Compute strides for given dimensions (row-major order)
    fn compute_strides(dims: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(dims.len());
        let mut stride = 1;

        for &dim in dims.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }

        strides.reverse();
        strides
    }
}

/// A GPU-based array that provides vector-like operations on AMD GPUs
/// Now supports multidimensional operations
pub struct ROCArray<T> {
    data: DeviceMemory<T>,
    shape: Shape,
    capacity: usize,
    _phantom: PhantomData<T>,
}

// Manual Debug implementation to avoid requiring Debug on T
impl<T> fmt::Debug for ROCArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ROCArray")
            .field("shape", &self.shape)
            .field("capacity", &self.capacity)
            .field("type", &std::any::type_name::<T>())
            .finish()
    }
}

impl<T> ROCArray<T>
where
    T: Copy + Default + 'static,
{
    /// Create a new empty ROCArray with the specified capacity (1D)
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        let data = DeviceMemory::new(capacity)?;
        Ok(Self {
            data,
            shape: Shape::new_1d(0), // Start with 0 length
            capacity,
            _phantom: PhantomData,
        })
    }

    /// Create a new ROCArray with specified shape
    pub fn new(shape: Shape) -> Result<Self> {
        let total_size = shape.size();
        let data = DeviceMemory::new(total_size)?;
        Ok(Self {
            data,
            shape,
            capacity: total_size,
            _phantom: PhantomData,
        })
    }

    /// Create a new 1D ROCArray
    pub fn new_1d(len: usize) -> Result<Self> {
        Self::new(Shape::new_1d(len))
    }

    /// Create a new 2D ROCArray (matrix)
    pub fn new_2d(rows: usize, cols: usize) -> Result<Self> {
        Self::new(Shape::new_2d(rows, cols))
    }

    /// Create a new 3D ROCArray
    pub fn new_3d(depth: usize, rows: usize, cols: usize) -> Result<Self> {
        Self::new(Shape::new_3d(depth, rows, cols))
    }

    /// Create a new ROCArray from host data
    pub fn from_vec(vec: Vec<T>) -> Result<Self> {
        let capacity = vec.len();
        let shape = Shape::new_1d(vec.len());
        let mut data = DeviceMemory::new(capacity)?;

        // Ensure data is copied before creating ROCArray
        data.copy_from_host(&vec)?;

        Ok(Self {
            data,
            shape,
            capacity,
            _phantom: PhantomData,
        })
    }

    /// Create a new ROCArray from host data with specific shape
    pub fn from_vec_with_shape(vec: Vec<T>, shape: Shape) -> Result<Self> {
        if vec.len() != shape.size() {
            return Err(crate::error::custom_error(format!(
                "Vector length {} doesn't match shape size {}",
                vec.len(),
                shape.size()
            )));
        }

        let mut data = DeviceMemory::new(vec.len())?;
        data.copy_from_host(&vec)?;

        Ok(Self {
            data,
            shape,
            capacity: vec.len(),
            _phantom: PhantomData,
        })
    }

    /// Create a new ROCArray filled with zeros
    pub fn zeros(shape: Shape) -> Result<Self> {
        let mut array = Self::new(shape)?;
        array.data.memset(0)?;
        Ok(array)
    }

    /// Create a new ROCArray filled with ones
    pub fn ones(shape: Shape) -> Result<Self>
    where
        T: From<u8>,
    {
        let one = T::from(1u8);
        let host_data = vec![one; shape.size()];
        Self::from_vec_with_shape(host_data, shape)
    }

    /// Create a new ROCArray filled with a specific value
    pub fn filled(shape: Shape, value: T) -> Result<Self>
    where
        T: Clone,
    {
        let host_data = vec![value; shape.size()];
        Self::from_vec_with_shape(host_data, shape)
    }

    // Shape and dimension methods

    /// Get the shape of the array
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the length of the array (total number of elements)
    pub fn len(&self) -> usize {
        self.shape.size()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get the capacity of the array
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the size of each dimension
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Reshape the array (must have same total size)
    pub fn reshape(&mut self, new_dims: Vec<usize>) -> Result<()> {
        let new_size: usize = new_dims.iter().product();
        if new_size != self.len() {
            return Err(crate::error::custom_error(
                "New shape must have the same total size".to_string(),
            ));
        }
        self.shape = Shape::new(new_dims);
        Ok(())
    }

    /// Create a reshaped view of the array
    pub fn reshaped(&self, new_dims: Vec<usize>) -> Result<ROCArray<T>> {
        let new_size: usize = new_dims.iter().product();
        if new_size != self.len() {
            return Err(crate::error::custom_error(
                "New shape must have the same total size".to_string(),
            ));
        }

        // In practice, you'd want to create a view that shares the same memory
        // For now, we'll create a copy
        let mut result = Self::new(Shape::new(new_dims))?;
        result.data.copy_from_device(&self.data)?;
        Ok(result)
    }

    /// Transpose the array (reverse all dimensions)
    pub fn transpose(&self) -> Result<ROCArray<T>>
    where
        T: kernels::TransposableOps,
    {
        let mut new_dims = self.shape.dims().to_vec();
        new_dims.reverse();
        let new_shape = Shape::new(new_dims);

        let mut result = ROCArray::new(new_shape.clone())?;
        kernels::transpose(&self.data, &result.data, &self.shape, &new_shape)?;
        Ok(result)
    }

    /// Squeeze dimensions of size 1
    pub fn squeeze(&mut self) {
        let squeezed_dims: Vec<usize> = self
            .shape
            .dims()
            .iter()
            .copied()
            .filter(|&d| d != 1)
            .collect();

        let new_dims = if squeezed_dims.is_empty() {
            vec![1]
        } else {
            squeezed_dims
        };

        self.shape = Shape::new(new_dims);
    }

    // Indexing and slicing methods

    /// Get element at specified indices
    pub fn get(&self, indices: &[usize]) -> Result<T> {
        let flat_index = self.shape.ravel_index(indices).ok_or_else(|| {
            crate::error::custom_error("Invalid indices for array shape".to_string())
        })?;

        kernels::get_element(&self.data, flat_index)
    }

    /// Set element at specified indices
    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        let flat_index = self.shape.ravel_index(indices).ok_or_else(|| {
            crate::error::custom_error("Invalid indices for array shape".to_string())
        })?;

        kernels::set_element(&mut self.data, flat_index, value)
    }

    /// Get a slice along the first dimension
    pub fn slice(&self, start: usize, end: usize) -> Result<ROCArray<T>> {
        if self.ndim() == 0 {
            return Err(crate::error::custom_error(
                "Cannot slice 0-dimensional array".to_string(),
            ));
        }

        let first_dim = self.shape.dims()[0];
        if start >= first_dim || end > first_dim || start >= end {
            return Err(crate::error::custom_error(
                "Invalid slice indices".to_string(),
            ));
        }

        let mut new_dims = self.shape.dims().to_vec();
        new_dims[0] = end - start;
        let new_shape = Shape::new(new_dims);

        let mut result = ROCArray::new(new_shape)?;
        kernels::slice_first_dim(&self.data, &result.data, &self.shape, start, end)?;
        Ok(result)
    }

    /// Get row (for 2D arrays)
    pub fn row(&self, index: usize) -> Result<ROCArray<T>> {
        if self.ndim() != 2 {
            return Err(crate::error::custom_error(
                "Row access requires 2D array".to_string(),
            ));
        }

        let row_data = self.slice(index, index + 1)?;
        let mut result = row_data;
        result.reshape(vec![self.shape.dims()[1]])?;
        Ok(result)
    }

    /// Get column (for 2D arrays)
    pub fn col(&self, index: usize) -> Result<ROCArray<T>> {
        if self.ndim() != 2 {
            return Err(crate::error::custom_error(
                "Column access requires 2D array".to_string(),
            ));
        }

        let mut result = ROCArray::new_1d(self.shape.dims()[0])?;
        kernels::extract_column(&self.data, &result.data, &self.shape, index)?;
        Ok(result)
    }

    // Data access methods

    /// Copy data to host
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut host_data = vec![T::default(); self.len()];
        self.data.copy_to_host(&mut host_data)?;
        Ok(host_data)
    }

    /// Copy data to host asynchronously
    pub fn to_vec_async(&self, stream: &Stream) -> crate::hip::error::Result<PendingCopy<T>> {
        let host_data = vec![T::default(); self.len()];
        self.data.copy_to_host_async(host_data, stream)
    }

    /// Get raw device pointer
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.data.as_ptr()
    }

    /// Get the underlying DeviceMemory reference
    pub fn device_memory(&self) -> &DeviceMemory<T> {
        &self.data
    }

    /// Copy from another ROCArray
    pub fn copy_from(&mut self, other: &ROCArray<T>) -> Result<()> {
        if other.len() > self.capacity {
            return Err(crate::error::custom_error(
                "Source array is larger than destination capacity".to_string(),
            ));
        }

        self.data.copy_from_device(&other.data)?;
        self.shape = other.shape.clone();
        Ok(())
    }

    /// Clone the array (creates a new ROCArray with copied data)
    pub fn clone_array(&self) -> Result<ROCArray<T>> {
        let mut new_array = ROCArray::new(self.shape.clone())?;
        new_array.copy_from(self)?;
        Ok(new_array)
    }
}

// Arithmetic operations with broadcasting support
impl<T> ROCArray<T>
where
    T: Copy + Default + 'static + kernels::NumericOps,
{
    /// Element-wise addition with broadcasting
    pub fn add(&self, other: &ROCArray<T>) -> Result<ROCArray<T>> {
        let result_shape = self.shape.broadcast_with(&other.shape).ok_or_else(|| {
            crate::error::custom_error("Shapes are not compatible for broadcasting".to_string())
        })?;

        let mut result = ROCArray::new(result_shape)?;

        if self.shape == other.shape {
            // Simple element-wise addition
            kernels::elementwise_add(&self.data, &other.data, &result.data, self.len())?;
        } else {
            // Addition with broadcasting
            kernels::elementwise_add_broadcast(
                &self.data,
                &other.data,
                &result.data,
                &self.shape,
                &other.shape,
                &result.shape,
            )?;
        }

        Ok(result)
    }

    /// Element-wise subtraction with broadcasting
    pub fn sub(&self, other: &ROCArray<T>) -> Result<ROCArray<T>> {
        let result_shape = self.shape.broadcast_with(&other.shape).ok_or_else(|| {
            crate::error::custom_error("Shapes are not compatible for broadcasting".to_string())
        })?;

        let mut result = ROCArray::new(result_shape)?;

        if self.shape == other.shape {
            kernels::elementwise_sub(&self.data, &other.data, &result.data, self.len())?;
        } else {
            kernels::elementwise_sub_broadcast(
                &self.data,
                &other.data,
                &result.data,
                &self.shape,
                &other.shape,
                &result.shape,
            )?;
        }

        Ok(result)
    }

    /// Element-wise multiplication with broadcasting
    pub fn mul(&self, other: &ROCArray<T>) -> Result<ROCArray<T>> {
        let result_shape = self.shape.broadcast_with(&other.shape).ok_or_else(|| {
            crate::error::custom_error("Shapes are not compatible for broadcasting".to_string())
        })?;

        let mut result = ROCArray::new(result_shape)?;

        if self.shape == other.shape {
            kernels::elementwise_mul(&self.data, &other.data, &result.data, self.len())?;
        } else {
            kernels::elementwise_mul_broadcast(
                &self.data,
                &other.data,
                &result.data,
                &self.shape,
                &other.shape,
                &result.shape,
            )?;
        }

        Ok(result)
    }

    /// Element-wise division with broadcasting
    pub fn div(&self, other: &ROCArray<T>) -> Result<ROCArray<T>> {
        let result_shape = self.shape.broadcast_with(&other.shape).ok_or_else(|| {
            crate::error::custom_error("Shapes are not compatible for broadcasting".to_string())
        })?;

        let mut result = ROCArray::new(result_shape)?;

        if self.shape == other.shape {
            kernels::elementwise_div(&self.data, &other.data, &result.data, self.len())?;
        } else {
            kernels::elementwise_div_broadcast(
                &self.data,
                &other.data,
                &result.data,
                &self.shape,
                &other.shape,
                &result.shape,
            )?;
        }

        Ok(result)
    }

    /// Scalar addition
    pub fn add_scalar(&self, scalar: T) -> Result<ROCArray<T>> {
        let mut result = ROCArray::new(self.shape.clone())?;
        kernels::scalar_add(&self.data, scalar, &result.data, self.len())?;
        Ok(result)
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: T) -> Result<ROCArray<T>> {
        let mut result = ROCArray::new(self.shape.clone())?;
        kernels::scalar_mul(&self.data, scalar, &result.data, self.len())?;
        Ok(result)
    }

    /// Matrix multiplication (only for 2D arrays)
    pub fn matmul(&self, other: &ROCArray<T>) -> Result<ROCArray<T>> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(crate::error::custom_error(
                "Matrix multiplication requires 2D arrays".to_string(),
            ));
        }

        let [m, k] = [self.shape.dims()[0], self.shape.dims()[1]];
        let [k2, n] = [other.shape.dims()[0], other.shape.dims()[1]];

        if k != k2 {
            return Err(crate::error::custom_error(
                "Inner dimensions must match for matrix multiplication".to_string(),
            ));
        }

        let result_shape = Shape::new_2d(m, n);
        let mut result = ROCArray::new(result_shape)?;

        kernels::matrix_multiply(&self.data, &other.data, &result.data, m, k, n)?;
        Ok(result)
    }

    /// Sum all elements
    pub fn sum(&self) -> Result<T> {
        kernels::reduce_sum(&self.data, self.len())
    }

    /// Sum along specified axis
    pub fn sum_axis(&self, axis: usize) -> Result<ROCArray<T>> {
        if axis >= self.ndim() {
            return Err(crate::error::custom_error("Axis out of bounds".to_string()));
        }

        let mut new_dims = self.shape.dims().to_vec();
        new_dims.remove(axis);
        let result_shape = if new_dims.is_empty() {
            Shape::new(vec![1])
        } else {
            Shape::new(new_dims)
        };

        let mut result = ROCArray::new(result_shape)?;
        kernels::reduce_sum_axis(&self.data, &result.data, &self.shape, axis)?;
        Ok(result)
    }

    /// Find maximum element
    pub fn max(&self) -> Result<T>
    where
        T: PartialOrd,
    {
        kernels::reduce_max(&self.data, self.len())
    }

    /// Find minimum element
    pub fn min(&self) -> Result<T>
    where
        T: PartialOrd,
    {
        kernels::reduce_min(&self.data, self.len())
    }

    /// Calculate mean
    pub fn mean(&self) -> Result<f64>
    where
        T: Into<f64>,
    {
        let sum: T = self.sum()?;
        Ok(sum.into() / self.len() as f64)
    }

    /// Mean along specified axis
    pub fn mean_axis(&self, axis: usize) -> Result<ROCArray<f64>>
    where
        T: Into<f64>,
    {
        let sum_result = self.sum_axis(axis)?;
        let axis_size = self.shape.dims()[axis] as f64;

        // Convert sum result to f64 and divide by axis size
        let sum_vec = sum_result.to_vec()?;
        let mean_vec: Vec<f64> = sum_vec.into_iter().map(|x| x.into() / axis_size).collect();

        ROCArray::from_vec_with_shape(mean_vec, sum_result.shape)
    }
}

// Random generation methods
impl<T> ROCArray<T>
where
    T: Copy + Default + 'static,
{
    /// Create ROCArray with random uniform values
    pub fn random_uniform(shape: Shape, seed: Option<u64>) -> Result<Self>
    where
        T: random::UniformRandom,
    {
        let mut array = Self::new(shape)?;
        let len = array.len();
        random::fill_uniform(&mut array.data, len, seed)?;
        Ok(array)
    }

    /// Create ROCArray with random normal values
    pub fn random_normal(shape: Shape, mean: f32, stddev: f32, seed: Option<u64>) -> Result<Self>
    where
        T: random::NormalRandom,
    {
        let mut array = Self::new(shape)?;
        let len = array.len();
        random::fill_normal(&mut array.data, len, mean, stddev, seed)?;
        Ok(array)
    }

    /// Fill with uniformly distributed random values
    pub fn fill_random_uniform(&mut self, seed: Option<u64>) -> Result<()>
    where
        T: random::UniformRandom,
    {
        let len = self.len();
        random::fill_uniform(&mut self.data, len, seed)
    }

    /// Fill with normally distributed random values
    pub fn fill_random_normal(&mut self, mean: f32, stddev: f32, seed: Option<u64>) -> Result<()>
    where
        T: random::NormalRandom,
    {
        let len = self.len();
        random::fill_normal(&mut self.data, len, mean, stddev, seed)
    }
}

// Sorting operations
impl<T> ROCArray<T>
where
    T: Copy + Default + 'static + sorting::Sortable,
{
    /// Sort array in ascending order
    pub fn sort(&mut self) -> Result<()> {
        let len = self.len();
        sorting::sort_ascending(&mut self.data, len)
    }

    /// Check if array is sorted
    pub fn is_sorted(&self) -> Result<bool> {
        sorting::is_sorted(&self.data, self.len())
    }

    /// Sort array in descending order
    pub fn sort_descending(&mut self) -> Result<()> {
        let len = self.len();
        sorting::sort_descending(&mut self.data, len)
    }

    /// Get indices that would sort the array (argsort)
    pub fn argsort(&self) -> Result<ROCArray<u32>> {
        let indices = ROCArray::<u32>::new_1d(self.len())?;
        sorting::argsort(&self.data, &indices.data, self.len())?;
        Ok(indices)
    }

    /// Partial sort (sort only the first k elements)
    pub fn partial_sort(&mut self, k: usize) -> Result<()> {
        let len = self.len();
        sorting::partial_sort(&mut self.data, len, k)
    }
}

// Async operations
impl<T> ROCArray<T>
where
    T: Copy + Default + 'static + kernels::NumericOps,
{
    /// Asynchronously add two arrays
    pub fn add_async(&self, other: &ROCArray<T>, stream: &Stream) -> Result<ROCArray<T>> {
        let result_shape = self.shape.broadcast_with(&other.shape).ok_or_else(|| {
            crate::error::custom_error("Shapes are not compatible for broadcasting".to_string())
        })?;

        let mut result = ROCArray::new(result_shape)?;

        if self.shape == other.shape {
            kernels::elementwise_add_async(
                &self.data,
                &other.data,
                &result.data,
                self.len(),
                stream,
            )?;
        } else {
            kernels::elementwise_add_broadcast_async(
                &self.data,
                &other.data,
                &result.data,
                &self.shape,
                &other.shape,
                &result.shape,
                stream,
            )?;
        }

        Ok(result)
    }
}

// Display implementation
impl<T> fmt::Display for ROCArray<T>
where
    T: Copy + Default + fmt::Debug + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_vec() {
            Ok(vec) => {
                match self.ndim() {
                    1 => {
                        if vec.len() <= 10 {
                            write!(f, "ROCArray{:?}", vec)
                        } else {
                            write!(
                                f,
                                "ROCArray[{:?}, …, {:?}] (len={})",
                                &vec[..3],
                                &vec[vec.len() - 3..],
                                vec.len()
                            )
                        }
                    }
                    2 => {
                        let [rows, cols] = [self.shape.dims()[0], self.shape.dims()[1]];
                        write!(f, "ROCArray2D({}x{})[\n", rows, cols)?;
                        for i in 0..rows.min(5) {
                            // Show max 5 rows
                            write!(f, "  [")?;
                            for j in 0..cols.min(5) {
                                // Show max 5 cols
                                let idx = i * cols + j;
                                if j > 0 {
                                    write!(f, ", ")?;
                                }
                                write!(f, "{:?}", vec[idx])?;
                            }
                            if cols > 5 {
                                write!(f, ", ...")?;
                            }
                            write!(f, "]\n")?;
                        }
                        if rows > 5 {
                            write!(f, "  ...\n")?;
                        }
                        write!(f, "]")
                    }
                    _ => write!(f, "ROCArray{}D{:?}", self.ndim(), self.shape.dims()),
                }
            }
            Err(_) => write!(f, "ROCArray{}D{:?}", self.ndim(), self.shape.dims()),
        }
    }
}

// Convenience type aliases
pub type ROCMatrix<T> = ROCArray<T>;
pub type ROCVector<T> = ROCArray<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multidim_creation() -> Result<()> {
        let arr = ROCArray::<f32>::new_2d(3, 4)?;
        assert_eq!(arr.shape().dims(), &[3, 4]);
        assert_eq!(arr.len(), 12);
        assert_eq!(arr.ndim(), 2);
        Ok(())
    }

    #[test]
    fn test_reshape() -> Result<()> {
        let mut arr = ROCArray::<f32>::new_1d(12)?;
        arr.reshape(vec![3, 4]);
        assert_eq!(arr.ndim(), 2);
        Ok(())
    }

    #[test]
    fn test_broadcasting_compatibility() {
        let shape1 = Shape::new(vec![3, 1, 4]);
        let shape2 = Shape::new(vec![1, 2, 4]);
        assert!(shape1.can_broadcast_with(&shape2));

        let result = shape1.broadcast_with(&shape2).unwrap();
        assert_eq!(result.dims(), &[3, 2, 4]);
    }
}
