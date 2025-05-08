extern crate core;
pub mod miopen;
pub mod rocblas;
pub mod rocfft;
pub mod rocrand;
pub mod rocsolver;

pub mod error;
pub mod hip;
use crate::rocfft::examples;

#[cfg(feature = "rocm_smi")]
pub mod rocmsmi;
pub mod rocsparse;
mod rocprofiler;

#[cfg(test)]
mod tests {
    use crate::hip;
    use crate::rocprofiler::types::{Feature, InfoData, Parameter, ParameterName, ProfilerMode};
    use crate::hip::{device_synchronize, get_device_count, Device, DeviceMemory};
    use crate::rocfft::examples::run_1d_complex_example;
    use crate::rocprofiler::context::Properties;
    use crate::rocprofiler::profiler::{get_metrics, version_string, Profiler};
    use crate::rocrand::utils::{generate_normal_f32, generate_uniform_f64};



    #[test]
    fn test_rocprofiler_simple() {
        // Check device availability first
        match hip::device_count() {
            Ok(count) => println!("Found {} device(s)", count),
            Err(e) => {
                println!("Error getting device count: {}", e);
                return;
            }
        }

        let device = match Device::new(0) {
            Ok(dev) => {
                let props = dev.properties().unwrap_or_else(|_| panic!("Couldn't get device properties"));
                println!("Using device: {}", props.name);
                dev
            },
            Err(e) => {
                println!("Failed to get device: {}", e);
                return;
            }
        };

        // Try with a single widely-supported metric
        let features = vec![
            Feature::new_metric("GRBM_COUNT", vec![
                Parameter::new(ParameterName::SeMask, 0xffffffff)
            ]),
        ];

        println!("Creating profiler...");
        let profiler = match Profiler::new(
            device,
            features,
            &[ProfilerMode::Standalone, ProfilerMode::SingleGroup],
            None
        ) {
            Ok(p) => {
                println!("Profiler created successfully");
                p
            },
            Err(e) => {
                println!("Failed to create profiler: {} (code: {})", e, e.code());
                return;
            }
        };
    }
}
