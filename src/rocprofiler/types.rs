// src/rocprofiler/types.rs

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::ptr;

use crate::hip;
use crate::rocprofiler::bindings;
use crate::rocprofiler::bindings::rocprofiler_feature_kind_t;
use crate::rocprofiler::error::{Error, Result};

/// Enumeration of different feature kinds that can be profiled
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    /// Performance metric
    Metric,
    /// Execution trace
    Trace,
    /// SPM Module
    SpmMod,
    /// PC Sample Module
    PcSmpMod,
}

impl FeatureKind {
    /// Convert to the native ROCProfiler feature kind
    pub fn to_native(&self) -> bindings::rocprofiler_feature_kind_t {
        match self {
            FeatureKind::Metric => bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_METRIC,
            FeatureKind::Trace => bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_TRACE,
            FeatureKind::SpmMod => bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_SPM_MOD,
            FeatureKind::PcSmpMod => bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_PCSMP_MOD,
        }
    }

    /// Convert from the native ROCProfiler feature kind
    pub fn from_native(kind: bindings::rocprofiler_feature_kind_t) -> Self {
        match kind {
            bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_METRIC => FeatureKind::Metric,
            bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_TRACE => FeatureKind::Trace,
            bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_SPM_MOD => FeatureKind::SpmMod,
            bindings::rocprofiler_feature_kind_t_ROCPROFILER_FEATURE_KIND_PCSMP_MOD => FeatureKind::PcSmpMod,
            _ => FeatureKind::Metric, // Default to metric for unknown types
        }
    }
}

/// Enumeration of different data kinds in profiling results
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataKind {
    /// Uninitialized data
    Uninit,
    /// 32-bit integer data
    Int32,
    /// 64-bit integer data
    Int64,
    /// Floating-point data
    Float,
    /// Double-precision data
    Double,
    /// Raw byte data
    Bytes,
}

impl DataKind {
    /// Convert to the native ROCProfiler data kind
    pub fn to_native(&self) -> bindings::rocprofiler_data_kind_t {
        match self {
            DataKind::Uninit => bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_UNINIT,
            DataKind::Int32 => bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_INT32,
            DataKind::Int64 => bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_INT64,
            DataKind::Float => bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_FLOAT,
            DataKind::Double => bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_DOUBLE,
            DataKind::Bytes => bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_BYTES,
        }
    }

    /// Convert from the native ROCProfiler data kind
    pub fn from_native(kind: bindings::rocprofiler_data_kind_t) -> Self {
        match kind {
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_UNINIT => DataKind::Uninit,
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_INT32 => DataKind::Int32,
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_INT64 => DataKind::Int64,
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_FLOAT => DataKind::Float,
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_DOUBLE => DataKind::Double,
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_BYTES => DataKind::Bytes,
            _ => DataKind::Uninit, // Default to uninit for unknown types
        }
    }
}

/// Represents a parameter for a profiling feature
#[derive(Debug, Clone)]
pub struct Parameter {
    name: ParameterName,
    value: u32,
}

/// Enumeration of parameter names
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterName {
    /// Select the target compute unit for profiling
    ComputeUnitTarget,
    /// VMID Mask
    VmIdMask,
    /// Legacy mask (deprecated)
    Mask,
    /// Legacy token mask (deprecated)
    TokenMask,
    /// Legacy token mask2 (deprecated)
    TokenMask2,
    /// Shader engine mask for selection
    SeMask,
    /// Legacy sample rate (deprecated)
    SampleRate,
    /// Legacy K concurrent (deprecated)
    KConcurrent,
    /// Set SIMD Mask or SIMD ID for collection
    SimdSelection,
    /// Set true for occupancy collection only
    OccupancyMode,
    /// ATT collection max data size (MB)
    AttBufferSize,
    /// Custom parameter name by value
    Custom(u32),
}

impl ParameterName {
    /// Convert to the native ROCProfiler parameter name
    pub fn to_native(&self) -> bindings::hsa_ven_amd_aqlprofile_parameter_name_t {
        match self {
            ParameterName::ComputeUnitTarget =>
                bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET,
            ParameterName::VmIdMask =>
                bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK,
            ParameterName::Mask =>
                bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK,
            ParameterName::TokenMask =>
                bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK,
            ParameterName::TokenMask2 =>
                bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2,
            ParameterName::SeMask =>
                bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK,
            ParameterName::SampleRate =>
                bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SAMPLE_RATE,
            ParameterName::KConcurrent =>
                bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT,
            ParameterName::SimdSelection =>
                bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION,
            ParameterName::OccupancyMode =>
                bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE,
            ParameterName::AttBufferSize =>
                bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE,
            ParameterName::Custom(val) => *val,
        }
    }

    /// Convert from the native ROCProfiler parameter name
    pub fn from_native(name: bindings::hsa_ven_amd_aqlprofile_parameter_name_t) -> Self {
        match name {
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET =>
                ParameterName::ComputeUnitTarget,
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK =>
                ParameterName::VmIdMask,
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK =>
                ParameterName::Mask,
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK =>
                ParameterName::TokenMask,
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2 =>
                ParameterName::TokenMask2,
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK =>
                ParameterName::SeMask,
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SAMPLE_RATE =>
                ParameterName::SampleRate,
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT =>
                ParameterName::KConcurrent,
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION =>
                ParameterName::SimdSelection,
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE =>
                ParameterName::OccupancyMode,
            bindings::hsa_ven_amd_aqlprofile_parameter_name_t_HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE =>
                ParameterName::AttBufferSize,
            _ => ParameterName::Custom(name),
        }
    }
}

impl Parameter {
    /// Create a new parameter with the given name and value
    pub fn new(name: ParameterName, value: u32) -> Self {
        Self { name, value }
    }

    /// Get parameter name
    pub fn name(&self) -> ParameterName {
        self.name
    }

    /// Get parameter value
    pub fn value(&self) -> u32 {
        self.value
    }

    /// Convert to the native ROCProfiler parameter
    pub fn to_native(&self) -> bindings::rocprofiler_parameter_t {
        bindings::rocprofiler_parameter_t {
            parameter_name: self.name.to_native(),
            value: self.value,
        }
    }

    /// Convert from the native ROCProfiler parameter
    pub fn from_native(param: &bindings::rocprofiler_parameter_t) -> Self {
        Self {
            name: ParameterName::from_native(param.parameter_name),
            value: param.value,
        }
    }
}

/// Represents the result data from profiling
#[derive(Debug, Clone)]
pub enum Data {
    /// Uninitialized data
    Uninit,
    /// 32-bit integer result
    Int32(u32),
    /// 64-bit integer result
    Int64(u64),
    /// Floating-point result
    Float(f32),
    /// Double-precision result
    Double(f64),
    /// Raw byte data
    Bytes(Vec<u8>, u32),
}

impl Data {
    /// Get the kind of this data
    pub fn kind(&self) -> DataKind {
        match self {
            Data::Uninit => DataKind::Uninit,
            Data::Int32(_) => DataKind::Int32,
            Data::Int64(_) => DataKind::Int64,
            Data::Float(_) => DataKind::Float,
            Data::Double(_) => DataKind::Double,
            Data::Bytes(_, _) => DataKind::Bytes,
        }
    }

    /// Convert from the native ROCProfiler data
    pub unsafe fn from_native(data: &bindings::rocprofiler_data_t) -> Self {
        match data.kind {
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_INT32 => {
                Data::Int32(data.__bindgen_anon_1.result_int32)
            }
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_INT64 => {
                Data::Int64(data.__bindgen_anon_1.result_int64)
            }
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_FLOAT => {
                Data::Float(data.__bindgen_anon_1.result_float)
            }
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_DOUBLE => {
                Data::Double(data.__bindgen_anon_1.result_double)
            }
            bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_BYTES => {
                let bytes_data = &data.__bindgen_anon_1.result_bytes;
                if bytes_data.ptr.is_null() || bytes_data.size == 0 {
                    Data::Bytes(Vec::new(), bytes_data.instance_count)
                } else {
                    let slice = std::slice::from_raw_parts(
                        bytes_data.ptr as *const u8,
                        bytes_data.size as usize,
                    );
                    Data::Bytes(slice.to_vec(), bytes_data.instance_count)
                }
            }
            _ => Data::Uninit,
        }
    }
}

/// Represents a profiling feature, which can be a counter or a metric
#[derive(Debug, Clone)]
pub enum Feature {
    /// Named metric feature
    Metric {
        /// Name of the metric
        name: String,
        /// Parameters for the metric
        parameters: Vec<Parameter>,
        /// Result data
        data: Option<Data>,
    },
    /// Hardware counter feature
    Counter {
        /// Block name
        block: String,
        /// Event ID
        event: u32,
        /// Parameters for the counter
        parameters: Vec<Parameter>,
        /// Result data
        data: Option<Data>,
    },
}

impl Feature {
    /// Create a new metric feature
    pub fn new_metric<S: Into<String>>(name: S, parameters: Vec<Parameter>) -> Self {
        Feature::Metric {
            name: name.into(),
            parameters,
            data: None,
        }
    }

    /// Create a new counter feature
    pub fn new_counter<S: Into<String>>(block: S, event: u32, parameters: Vec<Parameter>) -> Self {
        Feature::Counter {
            block: block.into(),
            event,
            parameters,
            data: None,
        }
    }

    /// Get the kind of this feature
    pub fn kind(&self) -> FeatureKind {
        match self {
            Feature::Metric { .. } => FeatureKind::Metric,
            Feature::Counter { .. } => FeatureKind::Metric, // In ROCProfiler, counters are also kind=METRIC
        }
    }

    /// Get the name of this feature
    pub fn name(&self) -> String {
        match self {
            Feature::Metric { name, .. } => name.clone(),
            Feature::Counter { block, event, .. } => format!("{}:0x{:x}", block, event),
        }
    }

    /// Get the parameters for this feature
    pub fn parameters(&self) -> &Vec<Parameter> {
        match self {
            Feature::Metric { parameters, .. } => parameters,
            Feature::Counter { parameters, .. } => parameters,
        }
    }

    /// Get the data for this feature, if available
    pub fn data(&self) -> Option<&Data> {
        match self {
            Feature::Metric { data, .. } => data.as_ref(),
            Feature::Counter { data, .. } => data.as_ref(),
        }
    }

    /// Update the feature with data
    pub unsafe fn update_with_data(&mut self, data: Data) {
        match self {
            Feature::Metric { data: d, .. } => *d = Some(data),
            Feature::Counter { data: d, .. } => *d = Some(data),
        }
    }

    /// Convert to a native ROCProfiler feature (for sending to the ROCProfiler API)
    pub fn to_native(&self) -> Result<(bindings::rocprofiler_feature_t, Vec<CString>, Vec<bindings::rocprofiler_parameter_t>)> {
        let (kind, name_ptr, counter_block, counter_event) = match self {
            Feature::Metric { name, .. } => {
                let name_cstr = CString::new(name.clone())
                    .map_err(|_| Error::new(bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT))?;
                (
                    FeatureKind::Metric.to_native(),
                    name_cstr,
                    CString::new("").unwrap(), // Not used for metrics
                    0u32,
                )
            }
            Feature::Counter { block, event, .. } => {
                let block_cstr = CString::new(block.clone())
                    .map_err(|_| Error::new(bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT))?;
                (
                    FeatureKind::Metric.to_native(),
                    CString::new("").unwrap(), // Not used for counters
                    block_cstr,
                    *event,
                )
            }
        };

        // Convert parameters
        let params = match self {
            Feature::Metric { parameters, .. } | Feature::Counter { parameters, .. } => {
                parameters.iter().map(|p| p.to_native()).collect::<Vec<_>>()
            }
        };

        // Save string pointers to ensure they live long enough
        let mut string_ptrs = Vec::new();
        if matches!(self, Feature::Metric { .. }) {
            string_ptrs.push(name_ptr);
        } else {
            string_ptrs.push(counter_block);
        }

        // Create the native feature structure
        let mut native_feature = bindings::rocprofiler_feature_t {
            kind,
            __bindgen_anon_1: unsafe {
                if matches!(self, Feature::Metric { .. }) {
                    let mut u: bindings::rocprofiler_feature_t__bindgen_ty_1 = std::mem::zeroed();
                    u.name = string_ptrs[0].as_ptr();
                    u
                } else {
                    let mut u: bindings::rocprofiler_feature_t__bindgen_ty_1 = std::mem::zeroed();
                    u.counter.block = string_ptrs[0].as_ptr();
                    u.counter.event = counter_event;
                    u
                }
            },
            parameters: if params.is_empty() {
                ptr::null()
            } else {
                params.as_ptr()
            },
            parameter_count: params.len() as u32,
            data: unsafe { std::mem::zeroed() },
        };

        // Set the data kind to uninit
        native_feature.data.kind = bindings::rocprofiler_data_kind_t_ROCPROFILER_DATA_KIND_UNINIT;

        Ok((native_feature, string_ptrs, params))
    }

    /// Update this feature from a native ROCProfiler feature
    pub unsafe fn update_from_native(&mut self, native: &bindings::rocprofiler_feature_t) {
        let data = Data::from_native(&native.data);
        self.update_with_data(data);
    }
}

/// Represents profiling modes for a context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilerMode {
    /// Standalone mode when ROC profiler supports a queue
    Standalone,
    /// ROC profiler creates queue in standalone mode
    CreateQueue,
    /// Only one group is allowed, failed otherwise
    SingleGroup,
}

impl ProfilerMode {
    /// Convert to the native ROCProfiler mode
    pub fn to_native(&self) -> bindings::rocprofiler_mode_t {
        match self {
            ProfilerMode::Standalone => bindings::rocprofiler_mode_t_ROCPROFILER_MODE_STANDALONE,
            ProfilerMode::CreateQueue => bindings::rocprofiler_mode_t_ROCPROFILER_MODE_CREATEQUEUE,
            ProfilerMode::SingleGroup => bindings::rocprofiler_mode_t_ROCPROFILER_MODE_SINGLEGROUP,
        }
    }

    /// Convert from the native ROCProfiler mode
    pub fn from_native(mode: bindings::rocprofiler_mode_t) -> Vec<ProfilerMode> {
        let mut modes = Vec::new();

        if mode & bindings::rocprofiler_mode_t_ROCPROFILER_MODE_STANDALONE != 0 {
            modes.push(ProfilerMode::Standalone);
        }

        if mode & bindings::rocprofiler_mode_t_ROCPROFILER_MODE_CREATEQUEUE != 0 {
            modes.push(ProfilerMode::CreateQueue);
        }

        if mode & bindings::rocprofiler_mode_t_ROCPROFILER_MODE_SINGLEGROUP != 0 {
            modes.push(ProfilerMode::SingleGroup);
        }

        modes
    }

    /// Combine multiple modes into a single mode value
    pub fn combine(modes: &[ProfilerMode]) -> u32 {
        let mut result = 0;

        for mode in modes {
            result |= mode.to_native();
        }

        result
    }
}

/// Represents a group of profiling features
// src/rocprofiler/types.rs (continued)

/// Represents a group of profiling features
#[derive(Debug)]
pub struct Group<'a> {
    group: bindings::rocprofiler_group_t,
    phantom: PhantomData<&'a ()>,
}

impl<'a> Group<'a> {
    /// Create a new group from a native ROCProfiler group
    pub fn from_native(group: bindings::rocprofiler_group_t) -> Self {
        Self {
            group,
            phantom: PhantomData,
        }
    }

    /// Get the group's index
    pub fn index(&self) -> u32 {
        self.group.index as u32
    }

    /// Get the native group handle
    pub fn as_native(&self) -> &bindings::rocprofiler_group_t {
        &self.group
    }

    /// Start profiling this group
    pub fn start(&self) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_group_start(&mut self.group.clone()) };
        Error::from_rocprofiler_error(status)
    }

    /// Stop profiling this group
    pub fn stop(&self) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_group_stop(&mut self.group.clone()) };
        Error::from_rocprofiler_error(status)
    }

    /// Read profiling data for this group
    pub fn read(&self) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_group_read(&mut self.group.clone()) };
        Error::from_rocprofiler_error(status)
    }

    /// Get profiling data for this group
    pub fn get_data(&self) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_group_get_data(&mut self.group.clone()) };
        Error::from_rocprofiler_error(status)
    }
}

/// HSA event ID enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HsaEvtId {
    /// Memory allocation event
    Allocate,
    /// Device assignment event
    Device,
    /// Memory copy event
    Memcopy,
    /// Packet submission event
    Submit,
    /// Kernel symbol event
    Ksymbol,
    /// Code object event
    Codeobj,
    /// Number of event types
    Number,
}

impl HsaEvtId {
    /// Convert to the native HSA event ID
    pub fn to_native(&self) -> bindings::hsa_evt_id_t {
        match self {
            HsaEvtId::Allocate => bindings::hsa_evt_id_t_HSA_EVT_ID_ALLOCATE,
            HsaEvtId::Device => bindings::hsa_evt_id_t_HSA_EVT_ID_DEVICE,
            HsaEvtId::Memcopy => bindings::hsa_evt_id_t_HSA_EVT_ID_MEMCOPY,
            HsaEvtId::Submit => bindings::hsa_evt_id_t_HSA_EVT_ID_SUBMIT,
            HsaEvtId::Ksymbol => bindings::hsa_evt_id_t_HSA_EVT_ID_KSYMBOL,
            HsaEvtId::Codeobj => bindings::hsa_evt_id_t_HSA_EVT_ID_CODEOBJ,
            HsaEvtId::Number => bindings::hsa_evt_id_t_HSA_EVT_ID_NUMBER,
        }
    }

    /// Convert from the native HSA event ID
    pub fn from_native(evt_id: bindings::hsa_evt_id_t) -> Self {
        match evt_id {
            bindings::hsa_evt_id_t_HSA_EVT_ID_ALLOCATE => HsaEvtId::Allocate,
            bindings::hsa_evt_id_t_HSA_EVT_ID_DEVICE => HsaEvtId::Device,
            bindings::hsa_evt_id_t_HSA_EVT_ID_MEMCOPY => HsaEvtId::Memcopy,
            bindings::hsa_evt_id_t_HSA_EVT_ID_SUBMIT => HsaEvtId::Submit,
            bindings::hsa_evt_id_t_HSA_EVT_ID_KSYMBOL => HsaEvtId::Ksymbol,
            bindings::hsa_evt_id_t_HSA_EVT_ID_CODEOBJ => HsaEvtId::Codeobj,
            bindings::hsa_evt_id_t_HSA_EVT_ID_NUMBER => HsaEvtId::Number,
            _ => HsaEvtId::Number,
        }
    }
}

// HSA event data wrappers

/// Allocation event data
#[derive(Debug, Clone)]
pub struct AllocateEventData {
    /// Allocated area pointer
    pub ptr: *const std::ffi::c_void,
    /// Allocated area size
    pub size: usize,
    /// Memory segment type
    pub segment: bindings::hsa_amd_segment_t,
    /// Memory global flag
    pub global_flag: bindings::hsa_amd_memory_pool_global_flag_t,
    /// Whether this is code allocation
    pub is_code: bool,
}

/// Device event data
#[derive(Debug, Clone)]
pub struct DeviceEventData {
    /// Device type
    pub device_type: bindings::hsa_device_type_t,
    /// Device ID
    pub id: u32,
    /// Device HSA agent
    pub agent: bindings::hsa_agent_t,
    /// Pointer the device is assigned to
    pub ptr: *const std::ffi::c_void,
}

/// Memory copy event data
#[derive(Debug, Clone)]
pub struct MemcopyEventData {
    /// Destination pointer
    pub dst: *const std::ffi::c_void,
    /// Source pointer
    pub src: *const std::ffi::c_void,
    /// Copy size
    pub size: usize,
}

/// Submit event data
#[derive(Debug, Clone)]
pub struct SubmitEventData {
    /// Packet pointer
    pub packet: *const std::ffi::c_void,
    /// Kernel name (if dispatch)
    pub kernel_name: Option<String>,
    /// HSA queue
    pub queue: *mut bindings::hsa_queue_t,
    /// Device type
    pub device_type: u32,
    /// Device ID
    pub device_id: u32,
}

/// Kernel symbol event data
#[derive(Debug, Clone)]
pub struct KsymbolEventData {
    /// Symbol object
    pub object: u64,
    /// Symbol name
    pub name: String,
    /// Name length
    pub name_length: u32,
    /// Symbol unload flag
    pub unload: bool,
}

/// Code object event data
#[derive(Debug, Clone)]
pub struct CodeobjEventData {
    /// Storage type
    pub storage_type: u32,
    /// Origin file descriptor
    pub storage_file: i32,
    /// Origin memory base
    pub memory_base: u64,
    /// Origin memory size
    pub memory_size: u64,
    /// Load base
    pub load_base: u64,
    /// Load size
    pub load_size: u64,
    /// Load delta
    pub load_delta: u64,
    /// URI string
    pub uri: Option<String>,
    /// Unload flag
    pub unload: bool,
}

/// HSA event data (union of all event types)
#[derive(Debug, Clone)]
pub enum HsaEventData {
    /// Allocation event
    Allocate(AllocateEventData),
    /// Device event
    Device(DeviceEventData),
    /// Memory copy event
    Memcopy(MemcopyEventData),
    /// Submit event
    Submit(SubmitEventData),
    /// Kernel symbol event
    Ksymbol(KsymbolEventData),
    /// Code object event
    Codeobj(CodeobjEventData),
}

impl HsaEventData {
    /// Create from a native HSA event data structure and event ID
    pub unsafe fn from_native(id: HsaEvtId, data: &bindings::hsa_evt_data_t) -> Self {
        match id {
            HsaEvtId::Allocate => {
                let allocate = &data.__bindgen_anon_1.allocate;
                HsaEventData::Allocate(AllocateEventData {
                    ptr: allocate.ptr,
                    size: allocate.size,
                    segment: allocate.segment,
                    global_flag: allocate.global_flag,
                    is_code: allocate.is_code != 0,
                })
            }
            HsaEvtId::Device => {
                let device = &data.__bindgen_anon_1.device;
                HsaEventData::Device(DeviceEventData {
                    device_type: device.type_,
                    id: device.id,
                    agent: device.agent,
                    ptr: device.ptr,
                })
            }
            HsaEvtId::Memcopy => {
                let memcopy = &data.__bindgen_anon_1.memcopy;
                HsaEventData::Memcopy(MemcopyEventData {
                    dst: memcopy.dst,
                    src: memcopy.src,
                    size: memcopy.size,
                })
            }
            HsaEvtId::Submit => {
                let submit = &data.__bindgen_anon_1.submit;
                let kernel_name = if !submit.kernel_name.is_null() {
                    let c_str = CStr::from_ptr(submit.kernel_name);
                    Some(c_str.to_string_lossy().into_owned())
                } else {
                    None
                };

                HsaEventData::Submit(SubmitEventData {
                    packet: submit.packet,
                    kernel_name,
                    queue: submit.queue,
                    device_type: submit.device_type,
                    device_id: submit.device_id,
                })
            }
            HsaEvtId::Ksymbol => {
                let ksymbol = &data.__bindgen_anon_1.ksymbol;
                let name = if !ksymbol.name.is_null() {
                    let c_str = CStr::from_ptr(ksymbol.name);
                    c_str.to_string_lossy().into_owned()
                } else {
                    String::new()
                };

                HsaEventData::Ksymbol(KsymbolEventData {
                    object: ksymbol.object,
                    name,
                    name_length: ksymbol.name_length,
                    unload: ksymbol.unload != 0,
                })
            }
            HsaEvtId::Codeobj => {
                let codeobj = &data.__bindgen_anon_1.codeobj;
                let uri = if !codeobj.uri.is_null() {
                    let c_str = CStr::from_ptr(codeobj.uri);
                    Some(c_str.to_string_lossy().into_owned())
                } else {
                    None
                };

                HsaEventData::Codeobj(CodeobjEventData {
                    storage_type: codeobj.storage_type,
                    storage_file: codeobj.storage_file,
                    memory_base: codeobj.memory_base,
                    memory_size: codeobj.memory_size,
                    load_base: codeobj.load_base,
                    load_size: codeobj.load_size,
                    load_delta: codeobj.load_delta,
                    uri,
                    unload: codeobj.unload != 0,
                })
            }
            _ => HsaEventData::Allocate(AllocateEventData {
                ptr: std::ptr::null(),
                size: 0,
                segment: 0,
                global_flag: 0,
                is_code: false,
            }),
        }
    }
}

/// Represents the settings for ROCProfiler
#[derive(Debug, Clone)]
pub struct Settings {
    /// Intercept mode
    pub intercept_mode: u32,
    /// Code object tracking
    pub code_obj_tracking: u32,
    /// Memory copy tracking
    pub memcopy_tracking: u32,
    /// Trace size
    pub trace_size: u32,
    /// Trace local
    pub trace_local: u32,
    /// Timeout
    pub timeout: u64,
    /// Timestamp on
    pub timestamp_on: u32,
    /// HSA intercepting
    pub hsa_intercepting: u32,
    /// K concurrent
    pub k_concurrent: u32,
    /// Opt mode
    pub opt_mode: u32,
    /// Object dumping
    pub obj_dumping: u32,
}

impl Settings {
    /// Create default settings
    pub fn new() -> Self {
        Self {
            intercept_mode: 0,
            code_obj_tracking: 0,
            memcopy_tracking: 0,
            trace_size: 0,
            trace_local: 0,
            timeout: 0,
            timestamp_on: 0,
            hsa_intercepting: 0,
            k_concurrent: 0,
            opt_mode: 0,
            obj_dumping: 0,
        }
    }

    /// Convert to native ROCProfiler settings
    pub fn to_native(&self) -> bindings::rocprofiler_settings_t {
        bindings::rocprofiler_settings_t {
            intercept_mode: self.intercept_mode,
            code_obj_tracking: self.code_obj_tracking,
            memcopy_tracking: self.memcopy_tracking,
            trace_size: self.trace_size,
            trace_local: self.trace_local,
            timeout: self.timeout,
            timestamp_on: self.timestamp_on,
            hsa_intercepting: self.hsa_intercepting,
            k_concurrent: self.k_concurrent,
            opt_mode: self.opt_mode,
            obj_dumping: self.obj_dumping,
        }
    }

    /// Convert from native ROCProfiler settings
    pub fn from_native(settings: &bindings::rocprofiler_settings_t) -> Self {
        Self {
            intercept_mode: settings.intercept_mode,
            code_obj_tracking: settings.code_obj_tracking,
            memcopy_tracking: settings.memcopy_tracking,
            trace_size: settings.trace_size,
            trace_local: settings.trace_local,
            timeout: settings.timeout,
            timestamp_on: settings.timestamp_on,
            hsa_intercepting: settings.hsa_intercepting,
            k_concurrent: settings.k_concurrent,
            opt_mode: settings.opt_mode,
            obj_dumping: settings.obj_dumping,
        }
    }
}

/// Time ID types for ROCProfiler
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeId {
    /// Linux realtime clock time
    ClockRealtime,
    /// Linux realtime-coarse clock time
    ClockRealtimeCoarse,
    /// Linux monotonic clock time
    ClockMonotonic,
    /// Linux monotonic-coarse clock time
    ClockMonotonicCoarse,
    /// Linux monotonic-raw clock time
    ClockMonotonicRaw,
}

impl TimeId {
    /// Convert to native ROCProfiler time ID
    pub fn to_native(&self) -> bindings::rocprofiler_time_id_t {
        match self {
            TimeId::ClockRealtime => bindings::rocprofiler_time_id_t_ROCPROFILER_TIME_ID_CLOCK_REALTIME,
            TimeId::ClockRealtimeCoarse => bindings::rocprofiler_time_id_t_ROCPROFILER_TIME_ID_CLOCK_REALTIME_COARSE,
            TimeId::ClockMonotonic => bindings::rocprofiler_time_id_t_ROCPROFILER_TIME_ID_CLOCK_MONOTONIC,
            TimeId::ClockMonotonicCoarse => bindings::rocprofiler_time_id_t_ROCPROFILER_TIME_ID_CLOCK_MONOTONIC_COARSE,
            TimeId::ClockMonotonicRaw => bindings::rocprofiler_time_id_t_ROCPROFILER_TIME_ID_CLOCK_MONOTONIC_RAW,
        }
    }

    /// Convert from native ROCProfiler time ID
    pub fn from_native(time_id: bindings::rocprofiler_time_id_t) -> Self {
        match time_id {
            bindings::rocprofiler_time_id_t_ROCPROFILER_TIME_ID_CLOCK_REALTIME => TimeId::ClockRealtime,
            bindings::rocprofiler_time_id_t_ROCPROFILER_TIME_ID_CLOCK_REALTIME_COARSE => TimeId::ClockRealtimeCoarse,
            bindings::rocprofiler_time_id_t_ROCPROFILER_TIME_ID_CLOCK_MONOTONIC => TimeId::ClockMonotonic,
            bindings::rocprofiler_time_id_t_ROCPROFILER_TIME_ID_CLOCK_MONOTONIC_COARSE => TimeId::ClockMonotonicCoarse,
            bindings::rocprofiler_time_id_t_ROCPROFILER_TIME_ID_CLOCK_MONOTONIC_RAW => TimeId::ClockMonotonicRaw,
            _ => TimeId::ClockRealtime,
        }
    }
}

/// Get the time value for a given time ID and profiling timestamp
pub fn get_time(time_id: TimeId, timestamp: u64) -> Result<(u64, u64)> {
    let mut value_ns = 0;
    let mut error_ns = 0;

    let status = unsafe {
        bindings::rocprofiler_get_time(
            time_id.to_native(),
            timestamp,
            &mut value_ns,
            &mut error_ns,
        )
    };

    if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
        return Err(Error::new(status));
    }

    Ok((value_ns, error_ns))
}

/// Information about a ROCProfiler metric
#[derive(Debug, Clone)]
pub struct MetricInfo {
    /// Agent index
    pub agent_index: u32,
    /// Metric name
    pub name: String,
    /// Number of instances
    pub instances: u32,
    /// Metric expression
    pub expr: Option<String>,
    /// Metric description
    pub description: Option<String>,
    /// Block name
    pub block_name: Option<String>,
    /// Number of block counters
    pub block_counters: u32,
}

/// Information about a ROCProfiler trace
#[derive(Debug, Clone)]
pub struct TraceInfo {
    /// Agent index
    pub agent_index: u32,
    /// Trace name
    pub name: String,
    /// Trace description
    pub description: Option<String>,
    /// Number of parameters
    pub parameter_count: u32,
}

/// Information about a ROCProfiler trace parameter
#[derive(Debug, Clone)]
pub struct TraceParameterInfo {
    /// Agent index
    pub agent_index: u32,
    /// Parameter code
    pub code: u32,
    /// Trace name
    pub trace_name: String,
    /// Parameter name
    pub parameter_name: String,
    /// Parameter description
    pub description: Option<String>,
}

/// Types of ROCProfiler info
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfoKind {
    /// Metric info
    Metric,
    /// Metric count
    MetricCount,
    /// Trace info
    Trace,
    /// Trace count
    TraceCount,
    /// Trace parameter info
    TraceParameter,
    /// Trace parameter count
    TraceParameterCount,
}

impl InfoKind {
    /// Convert to native ROCProfiler info kind
    pub fn to_native(&self) -> bindings::rocprofiler_info_kind_t {
        match self {
            InfoKind::Metric => bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_METRIC,
            InfoKind::MetricCount => bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_METRIC_COUNT,
            InfoKind::Trace => bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE,
            InfoKind::TraceCount => bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE_COUNT,
            InfoKind::TraceParameter => bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE_PARAMETER,
            InfoKind::TraceParameterCount => bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE_PARAMETER_COUNT,
        }
    }

    /// Convert from native ROCProfiler info kind
    pub fn from_native(kind: bindings::rocprofiler_info_kind_t) -> Self {
        match kind {
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_METRIC => InfoKind::Metric,
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_METRIC_COUNT => InfoKind::MetricCount,
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE => InfoKind::Trace,
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE_COUNT => InfoKind::TraceCount,
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE_PARAMETER => InfoKind::TraceParameter,
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE_PARAMETER_COUNT => InfoKind::TraceParameterCount,
            _ => InfoKind::Metric,
        }
    }
}

/// ROCProfiler info data
#[derive(Debug, Clone)]
pub enum InfoData {
    /// Metric info
    Metric(MetricInfo),
    /// Trace info
    Trace(TraceInfo),
    /// Trace parameter info
    TraceParameter(TraceParameterInfo),
}

impl InfoData {
    /// Create from native ROCProfiler info data
    pub unsafe fn from_native(info: &bindings::rocprofiler_info_data_t) -> Self {
        match info.kind {
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_METRIC => {
                let metric = &info.__bindgen_anon_1.metric;
                let name = if !metric.name.is_null() {
                    CStr::from_ptr(metric.name).to_string_lossy().into_owned()
                } else {
                    String::new()
                };

                let expr = if !metric.expr.is_null() {
                    Some(CStr::from_ptr(metric.expr).to_string_lossy().into_owned())
                } else {
                    None
                };

                let description = if !metric.description.is_null() {
                    Some(CStr::from_ptr(metric.description).to_string_lossy().into_owned())
                } else {
                    None
                };

                let block_name = if !metric.block_name.is_null() {
                    Some(CStr::from_ptr(metric.block_name).to_string_lossy().into_owned())
                } else {
                    None
                };

                InfoData::Metric(MetricInfo {
                    agent_index: info.agent_index,
                    name,
                    instances: metric.instances,
                    expr,
                    description,
                    block_name,
                    block_counters: metric.block_counters,
                })
            },
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE => {
                let trace = &info.__bindgen_anon_1.trace;
                let name = if !trace.name.is_null() {
                    CStr::from_ptr(trace.name).to_string_lossy().into_owned()
                } else {
                    String::new()
                };

                let description = if !trace.description.is_null() {
                    Some(CStr::from_ptr(trace.description).to_string_lossy().into_owned())
                } else {
                    None
                };

                InfoData::Trace(TraceInfo {
                    agent_index: info.agent_index,
                    name,
                    description,
                    parameter_count: trace.parameter_count,
                })
            },
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE_PARAMETER => {
                let param = &info.__bindgen_anon_1.trace_parameter;
                let trace_name = if !param.trace_name.is_null() {
                    CStr::from_ptr(param.trace_name).to_string_lossy().into_owned()
                } else {
                    String::new()
                };

                let parameter_name = if !param.parameter_name.is_null() {
                    CStr::from_ptr(param.parameter_name).to_string_lossy().into_owned()
                } else {
                    String::new()
                };

                let description = if !param.description.is_null() {
                    Some(CStr::from_ptr(param.description).to_string_lossy().into_owned())
                } else {
                    None
                };

                InfoData::TraceParameter(TraceParameterInfo {
                    agent_index: info.agent_index,
                    code: param.code,
                    trace_name,
                    parameter_name,
                    description,
                })
            },
            _ => {
                // Default to an empty metric
                InfoData::Metric(MetricInfo {
                    agent_index: info.agent_index,
                    name: String::new(),
                    instances: 0,
                    expr: None,
                    description: None,
                    block_name: None,
                    block_counters: 0,
                })
            }
        }
    }
}