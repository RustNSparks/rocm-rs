extern crate core;
pub mod error;
pub mod hip;
pub mod miopen;
pub mod rocblas;
pub mod rocfft;
pub mod rocrand;
pub mod rocsolver;

#[cfg(feature = "rocm_smi")]
pub mod rocmsmi;
mod rocprofiler;
pub mod rocsparse;

#[cfg(test)]
mod tests {
    use crate::hip;
    use crate::hip::Device;
    use crate::rocprofiler::profiler::Profiler;
    use crate::rocprofiler::types::{Feature, Parameter, ParameterName, ProfilerMode};

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
                let props = dev
                    .properties()
                    .unwrap_or_else(|_| panic!("Couldn't get device properties"));
                println!("Using device: {}", props.name);
                dev
            }
            Err(e) => {
                println!("Failed to get device: {}", e);
                return;
            }
        };

        // Try with a single widely-supported metric
        let features = vec![Feature::new_metric(
            "GRBM_COUNT",
            vec![Parameter::new(ParameterName::SeMask, 0xffffffff)],
        )];

        println!("Creating profiler...");
        let _profiler = match Profiler::new(
            device,
            features,
            &[ProfilerMode::Standalone, ProfilerMode::SingleGroup],
            None,
        ) {
            Ok(p) => {
                println!("Profiler created successfully");
                p
            }
            Err(e) => {
                println!("Failed to create profiler: {} (code: {})", e, e.code());
                return;
            }
        };

    }
}
