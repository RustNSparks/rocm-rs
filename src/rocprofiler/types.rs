// src/rocprofiler/types.rs

use crate::rocprofiler::error::{Error, Result};
use crate::rocprofiler::bindings;
use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;

/// Feature kinds for profiling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    /// Metric feature
    Metric = bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_METRIC as isize,
    /// Trace feature
    Trace = bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_TRACE as isize,
    /// SPM module
    SpmMod = bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_SPM_MOD as isize,
    /// PCSMP module
    PcsmpMod = bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_PCSMP_MOD as isize,
}

impl From<u32> for FeatureKind {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_METRIC => FeatureKind::Metric,
            bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_TRACE => FeatureKind::Trace,
            bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_SPM_MOD => FeatureKind::SpmMod,
            bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_PCSMP_MOD => FeatureKind::PcsmpMod,
            _ => FeatureKind::Metric, // Default case
        }
    }
}

/// Data kind for profiler data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataKind {
    /// Uninitialized data
    Uninit = bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_UNINIT as isize,
    /// 32-bit integer
    Int32 = bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_INT32 as isize,
    /// 64-bit integer
    Int64 = bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_INT64 as isize,
    /// 32-bit float
    Float = bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_FLOAT as isize,
    /// 64-bit float
    Double = bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_DOUBLE as isize,
    /// Byte array
    Bytes = bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_BYTES as isize,
}

impl From<u32> for DataKind {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_UNINIT => DataKind::Uninit,
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_INT32 => DataKind::Int32,
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_INT64 => DataKind::Int64,
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_FLOAT => DataKind::Float,
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_DOUBLE => DataKind::Double,
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_BYTES => DataKind::Bytes,
            _ => DataKind::Uninit, // Default case
        }
    }
}

/// Profiler modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilerMode {
    /// Standalone profiling mode
    Standalone = bindings::rocprofiler_mode_t_ROCPROFILER_MODE_STANDALONE as isize,
    /// Create queue profiling mode
    CreateQueue = bindings::rocprofiler_mode_t_ROCPROFILER_MODE_CREATEQUEUE as isize,
    /// Single group profiling mode
    SingleGroup = bindings::rocprofiler_mode_t_ROCPROFILER_MODE_SINGLEGROUP as isize,
}

/// Information kinds for querying profiler information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfoKind {
    /// Metric information
    Metric = bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_METRIC as isize,
    /// Metric count
    MetricCount = bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_METRIC_COUNT as isize,
    /// Trace information
    Trace = bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE as isize,
    /// Trace count
    TraceCount = bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE_COUNT as isize,
    /// Trace parameter information
    TraceParameter = bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE_PARAMETER as isize,
    /// Trace parameter count
    TraceParameterCount = bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE_PARAMETER_COUNT as isize,
}

/// Parameter names for feature configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterName {
    /// Shader engine mask
    SeMask = bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK as isize,
    /// Compute unit target
    ComputeUnitTarget = bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET as isize,
    /// SIMD selection
    SimdSelection = bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION as isize,
}

/// Parameter for feature configuration
#[derive(Debug, Clone)]
pub struct Parameter {
    parameter_name: u32,
    value: u32,
}

impl Parameter {
    /// Create a new parameter
    pub fn new(name: ParameterName, value: u32) -> Self {
        Self {
            parameter_name: name as u32,
            value,
        }
    }

    /// Get the parameter name
    pub fn name(&self) -> ParameterName {
        match self.parameter_name {
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK => ParameterName::SeMask,
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET => ParameterName::ComputeUnitTarget,
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION => ParameterName::SimdSelection,
            _ => ParameterName::SeMask, // Default case
        }
    }

    /// Get the parameter value
    pub fn value(&self) -> u32 {
        self.value
    }

    /// Convert to the C representation
    pub(crate) fn to_c(&self) -> bindings::rocprofiler_parameter_t {
        bindings::rocprofiler_parameter_t {
            parameter_name: self.parameter_name,
            value: self.value,
        }
    }
}

/// Data representation for profiling results
#[derive(Debug, Clone)]
pub enum Data {
    /// Uninitialized data
    Uninit,
    /// 32-bit integer data
    Int32(u32),
    /// 64-bit integer data
    Int64(u64),
    /// 32-bit float data
    Float(f32),
    /// 64-bit float data
    Double(f64),
    /// Byte array with instance count
    Bytes(Vec<u8>, u32),
}

impl Data {
    /// Convert from the C representation
    pub(crate) fn from_c(data: &bindings::rocprofiler_data_t) -> Self {
        match data.kind {
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_UNINIT => Data::Uninit,
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_INT32 => {
                // Safety: We know data.kind is INT32, so we can read result_int32
                let value = unsafe { data.__bindgen_anon_1.result_int32 };
                Data::Int32(value)
            },
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_INT64 => {
                // Safety: We know data.kind is INT64, so we can read result_int64
                let value = unsafe { data.__bindgen_anon_1.result_int64 };
                Data::Int64(value)
            },
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_FLOAT => {
                // Safety: We know data.kind is FLOAT, so we can read result_float
                let value = unsafe { data.__bindgen_anon_1.result_float };
                Data::Float(value)
            },
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_DOUBLE => {
                // Safety: We know data.kind is DOUBLE, so we can read result_double
                let value = unsafe { data.__bindgen_anon_1.result_double };
                Data::Double(value)
            },
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_BYTES => {
                // Safety: We know data.kind is BYTES, so we can read result_bytes
                unsafe {
                    let bytes_info = &data.__bindgen_anon_1.result_bytes;
                    if bytes_info.ptr.is_null() {
                        Data::Bytes(Vec::new(), 0)
                    } else {
                        let ptr = bytes_info.ptr as *const u8;
                        let size = bytes_info.size as usize;
                        let instances = bytes_info.instance_count;

                        // Copy the bytes to a new Vec
                        let data_slice = slice::from_raw_parts(ptr, size);
                        Data::Bytes(data_slice.to_vec(), instances)
                    }
                }
            },
            _ => Data::Uninit,
        }
    }
}

/// Metric information
#[derive(Debug, Clone)]
pub struct MetricInfo {
    /// Metric name
    pub name: String,
    /// Number of instances
    pub instances: u32,
    /// Expression (if any)
    pub expression: Option<String>,
    /// Description (if any)
    pub description: Option<String>,
    /// Block name (if any)
    pub block_name: Option<String>,
    /// Number of block counters
    pub block_counters: u32,
}

/// Trace information
#[derive(Debug, Clone)]
pub struct TraceInfo {
    /// Trace name
    pub name: String,
    /// Description (if any)
    pub description: Option<String>,
    /// Number of parameters
    pub parameter_count: u32,
}

/// Trace parameter information
#[derive(Debug, Clone)]
pub struct TraceParameterInfo {
    /// Parameter code
    pub code: u32,
    /// Trace name
    pub trace_name: String,
    /// Parameter name
    pub parameter_name: String,
    /// Description (if any)
    pub description: Option<String>,
}

/// Information data for querying profiler information
#[derive(Debug, Clone)]
pub enum InfoData {
    /// Metric information
    Metric(MetricInfo),
    /// Trace information
    Trace(TraceInfo),
    /// Trace parameter information
    TraceParameter(TraceParameterInfo),
}

// Implement conversion from the C representation
impl InfoData {
    /// Convert from the C representation
    pub(crate) fn from_c(data: &bindings::rocprofiler_info_data_t) -> Result<Self> {
        let kind = data.kind;

        match kind {
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_METRIC => {
                // Safety: We know data.kind is METRIC, so we can read metric data
                unsafe {
                    let metric_data = &data.__bindgen_anon_1.metric;

                    // Extract the name
                    let name = if metric_data.name.is_null() {
                        return Err(Error::new(bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT));
                    } else {
                        CStr::from_ptr(metric_data.name).to_string_lossy().into_owned()
                    };

                    // Extract optional fields
                    let expression = if metric_data.expr.is_null() {
                        None
                    } else {
                        Some(CStr::from_ptr(metric_data.expr).to_string_lossy().into_owned())
                    };

                    let description = if metric_data.description.is_null() {
                        None
                    } else {
                        Some(CStr::from_ptr(metric_data.description).to_string_lossy().into_owned())
                    };

                    let block_name = if metric_data.block_name.is_null() {
                        None
                    } else {
                        Some(CStr::from_ptr(metric_data.block_name).to_string_lossy().into_owned())
                    };

                    Ok(InfoData::Metric(MetricInfo {
                        name,
                        instances: metric_data.instances,
                        expression,
                        description,
                        block_name,
                        block_counters: metric_data.block_counters,
                    }))
                }
            },
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE => {
                // Safety: We know data.kind is TRACE, so we can read trace data
                unsafe {
                    let trace_data = &data.__bindgen_anon_1.trace;

                    // Extract the name
                    let name = if trace_data.name.is_null() {
                        return Err(Error::new(bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT));
                    } else {
                        CStr::from_ptr(trace_data.name).to_string_lossy().into_owned()
                    };

                    // Extract optional fields
                    let description = if trace_data.description.is_null() {
                        None
                    } else {
                        Some(CStr::from_ptr(trace_data.description).to_string_lossy().into_owned())
                    };

                    Ok(InfoData::Trace(TraceInfo {
                        name,
                        description,
                        parameter_count: trace_data.parameter_count,
                    }))
                }
            },
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE_PARAMETER => {
                // Safety: We know data.kind is TRACE_PARAMETER, so we can read trace parameter data
                unsafe {
                    let param_data = &data.__bindgen_anon_1.trace_parameter;

                    // Extract required fields
                    let trace_name = if param_data.trace_name.is_null() {
                        return Err(Error::new(bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT));
                    } else {
                        CStr::from_ptr(param_data.trace_name).to_string_lossy().into_owned()
                    };

                    let parameter_name = if param_data.parameter_name.is_null() {
                        return Err(Error::new(bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT));
                    } else {
                        CStr::from_ptr(param_data.parameter_name).to_string_lossy().into_owned()
                    };

                    // Extract optional fields
                    let description = if param_data.description.is_null() {
                        None
                    } else {
                        Some(CStr::from_ptr(param_data.description).to_string_lossy().into_owned())
                    };

                    Ok(InfoData::TraceParameter(TraceParameterInfo {
                        code: param_data.code,
                        trace_name,
                        parameter_name,
                        description,
                    }))
                }
            },
            _ => Err(Error::new(bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT)),
        }
    }
}

/// Group of features for profiling
#[derive(Debug)]
pub struct Group {
    index: u32,
    features: Vec<*mut bindings::rocprofiler_feature_t>,
    context: *mut bindings::rocprofiler_t,
}

impl Group {
    /// Create a new group from the C representation
    pub(crate) fn from_c(group: &bindings::rocprofiler_group_t) -> Self {
        let features_slice = unsafe {
            std::slice::from_raw_parts(group.features, group.feature_count as usize)
        };

        Self {
            index: group.index,
            features: features_slice.to_vec(),
            context: group.context,
        }
    }

    /// Get the group index
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Get the number of features in the group
    pub fn feature_count(&self) -> u32 {
        self.features.len() as u32
    }

    /// Start profiling for this group
    pub fn start(&self) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_group_start(&mut self.to_c()) };
        Error::from_hsa_status(status)
    }

    /// Stop profiling for this group
    pub fn stop(&self) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_group_stop(&mut self.to_c()) };
        Error::from_hsa_status(status)
    }

    /// Read profiling data for this group
    pub fn read(&self) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_group_read(&mut self.to_c()) };
        Error::from_hsa_status(status)
    }

    /// Get profiling data for this group
    pub fn get_data(&self) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_group_get_data(&mut self.to_c()) };
        Error::from_hsa_status(status)
    }

    /// Convert to the C representation
    pub(crate) fn to_c(&self) -> bindings::rocprofiler_group_t {
        bindings::rocprofiler_group_t {
            index: self.index,
            features: self.features.as_ptr() as *mut *mut bindings::rocprofiler_feature_t,
            feature_count: self.features.len() as u32,
            context: self.context,
        }
    }
}

/// Feature for profiling
pub struct Feature {
    kind: FeatureKind,
    name: String,
    name_cstr: Option<CString>,
    parameters: Vec<Parameter>,
    c_parameters: Vec<bindings::rocprofiler_parameter_t>,
    data: Option<Data>,
    c_feature: Option<Box<bindings::rocprofiler_feature_t>>,
}

impl Feature {
    /// Create a new metric feature
    pub fn new_metric<S: Into<String>>(name: S, parameters: Vec<Parameter>) -> Self {
        let name_string = name.into();
        let name_cstr = CString::new(name_string.clone()).unwrap();

        // Convert parameters to C representation
        let c_parameters: Vec<bindings::rocprofiler_parameter_t> = parameters
            .iter()
            .map(|p| p.to_c())
            .collect();

        Self {
            kind: FeatureKind::Metric,
            name: name_string,
            name_cstr: Some(name_cstr),
            parameters,
            c_parameters,
            data: None,
            c_feature: None,
        }
    }

    /// Get the feature name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the feature kind
    pub fn kind(&self) -> FeatureKind {
        self.kind
    }

    /// Get the feature data
    pub fn data(&self) -> Option<&Data> {
        self.data.as_ref()
    }

    /// Get the parameters
    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    // For internal use to update data after profiling
    pub(crate) fn set_data(&mut self, data: Data) {
        self.data = Some(data);
    }

    // For internal use to update data from C representation
    pub(crate) fn update_from_c(&mut self, c_feature: *mut bindings::rocprofiler_feature_t) {
        if !c_feature.is_null() {
            unsafe {
                self.data = Some(Data::from_c(&(*c_feature).data));
            }
        }
    }

    /// Convert to the C representation
    pub(crate) fn to_c(&mut self) -> *mut bindings::rocprofiler_feature_t {
        // If we already have a C feature, return it
        if let Some(feature) = &self.c_feature {
            return feature.as_ref() as *const bindings::rocprofiler_feature_t
                as *mut bindings::rocprofiler_feature_t;
        }

        // Otherwise, create a new C feature
        let name_ptr = self.name_cstr.as_ref()
            .map(|cstr| cstr.as_ptr())
            .unwrap_or(ptr::null());

        let mut feature = Box::new(bindings::rocprofiler_feature_t {
            kind: self.kind as u32,
            __bindgen_anon_1: bindings::rocprofiler_feature_t__bindgen_ty_1 {
                name: name_ptr,
            },
            parameters: if self.c_parameters.is_empty() {
                ptr::null()
            } else {
                self.c_parameters.as_ptr()
            },
            parameter_count: self.c_parameters.len() as u32,
            data: bindings::rocprofiler_data_t {
                kind: bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_UNINIT,
                __bindgen_anon_1: bindings::rocprofiler_data_t__bindgen_ty_1 {
                    result_int32: 0,
                },
            },
        });

        let ptr = feature.as_mut() as *mut bindings::rocprofiler_feature_t;
        self.c_feature = Some(feature);
        ptr
    }
}