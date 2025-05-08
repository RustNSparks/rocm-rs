// examples/rocprofiler_example.rs
//
// A simple example showing how to use the ROCProfiler API

use rocm_rs::hip::{Device, get_device_count};
use rocm_rs::rocprofiler::{
    Profiler, Feature, Parameter, ParameterName, InfoKind, ProfilerMode, Properties,
    get_metrics, get_metric_count, version_string, error_string
};

fn main() -> rocm_rs::rocprofiler::Result<()> {
    // Print ROCProfiler version
    println!("ROCProfiler version: {}", version_string());

    // Get the number of available devices
    let device_count = get_device_count().expect("Failed to get device count");
    if device_count == 0 {
        println!("No GPU devices found.");
        return Ok(());
    }

    // Get the first device
    let device = Device::new(0).expect("Failed to get device");
    println!("Using device: {}", device.properties()?.name);

    // Get the number of metrics
    let metric_count = get_metric_count(Some(&device))?;
    println!("Number of available metrics: {}", metric_count);

    // List available metrics
    let metrics = get_metrics(Some(&device))?;
    println!("\nAvailable metrics:");
    for metric in metrics {
        match metric {
            rocm_rs::rocprofiler::InfoData::Metric(info) => {
                println!("  {}: instances={}, block={:?}",
                         info.name,
                         info.instances,
                         info.block_name.unwrap_or_default()
                );
                if let Some(desc) = info.description {
                    println!("    Description: {}", desc);
                }
                if let Some(expr) = info.expr {
                    println!("    Expression: {}", expr);
                }
            },
            _ => {}
        }
    }

    // Create a set of features to profile
    let mut features = Vec::new();

    // Add a simple metric
    features.push(Feature::new_metric("GRBM_COUNT", vec![
        Parameter::new(ParameterName::SeMask, 0xffffffff),
    ]));

    // Add a counter
    features.push(Feature::new_counter("GRBM", 0x1, vec![
        Parameter::new(ParameterName::SeMask, 0xffffffff),
    ]));

    // Create profiler
    let properties = Properties::new()
        .with_queue_depth(16)
        .with_handler(|group| {
            println!("Group {} completed", group.index());
            true
        });

    let mut profiler = Profiler::new(
        device,
        features,
        &[ProfilerMode::Standalone, ProfilerMode::SingleGroup],
        Some(properties)
    )?;

    // Run the profiling session
    println!("\nStarting profiling...");
    profiler.profile_all()?;
    println!("Profiling completed.");

    // Display results
    // examples/rocprofiler_example.rs (continued)

    // Display results
    println!("\nResults:");
    for (i, feature) in profiler.features().iter().enumerate() {
        println!("Feature {}: {}", i, feature.name());

        if let Some(data) = feature.data() {
            match data {
                rocm_rs::rocprofiler::Data::Uninit => {
                    println!("  Data: Uninitialized");
                },
                rocm_rs::rocprofiler::Data::Int32(val) => {
                    println!("  Value (Int32): {}", val);
                },
                rocm_rs::rocprofiler::Data::Int64(val) => {
                    println!("  Value (Int64): {}", val);
                },
                rocm_rs::rocprofiler::Data::Float(val) => {
                    println!("  Value (Float): {}", val);
                },
                rocm_rs::rocprofiler::Data::Double(val) => {
                    println!("  Value (Double): {}", val);
                },
                rocm_rs::rocprofiler::Data::Bytes(bytes, instances) => {
                    println!("  Data: {} bytes, {} instances", bytes.len(), instances);

                    // If the data is small enough, print it
                    if bytes.len() <= 64 {
                        println!("  Bytes: {:?}", bytes);
                    }
                },
            }
        } else {
            println!("  No data available");
        }
    }

    // Demonstrate using queue callbacks
    println!("\nSetting up queue callbacks...");

    let callbacks = rocm_rs::rocprofiler::QueueCallbacks::new()
        .with_dispatch(|callback_data, group| {
            // Extract kernel name if available
            let kernel_name = if !callback_data.kernel_name.is_null() {
                let c_str = unsafe { std::ffi::CStr::from_ptr(callback_data.kernel_name) };
                Some(c_str.to_string_lossy().into_owned())
            } else {
                None
            };

            if let Some(name) = kernel_name {
                println!("Kernel dispatch: {}", name);
            } else {
                println!("Non-kernel dispatch");
            }

            Ok(())
        })
        .with_create(|queue| {
            println!("Queue created: {:?}", queue);
            Ok(())
        })
        .with_destroy(|queue| {
            println!("Queue destroyed: {:?}", queue);
            Ok(())
        });

    // Set the callbacks
    rocm_rs::rocprofiler::set_queue_callbacks(callbacks)?;

    // Start the callbacks
    rocm_rs::rocprofiler::start_queue_callbacks()?;

    println!("Queue callbacks are active. To see them in action, create and use a queue.");
    println!("Press Enter to continue...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).expect("Failed to read line");

    // Stop the callbacks
    rocm_rs::rocprofiler::stop_queue_callbacks()?;

    // Remove the callbacks
    rocm_rs::rocprofiler::remove_queue_callbacks()?;

    // Show how to use a profiled queue
    println!("\nCreating a profiled queue...");
    let profiled_queue = rocm_rs::rocprofiler::create_profiled_queue(
        device,
        256,  // queue size
        0,    // private segment size
        0,    // group segment size
    )?;

    println!("Profiled queue created: {:?}", profiled_queue);

    // Drop the queue
    drop(profiled_queue);

    // Example of how to handle HSA events
    println!("\nExample of handling HSA events:");
    println!("HsaEvtId::Allocate = {:?}", rocm_rs::rocprofiler::HsaEvtId::Allocate);
    println!("HsaEvtId::Device = {:?}", rocm_rs::rocprofiler::HsaEvtId::Device);
    println!("HsaEvtId::Memcopy = {:?}", rocm_rs::rocprofiler::HsaEvtId::Memcopy);
    println!("HsaEvtId::Submit = {:?}", rocm_rs::rocprofiler::HsaEvtId::Submit);
    println!("HsaEvtId::Ksymbol = {:?}", rocm_rs::rocprofiler::HsaEvtId::Ksymbol);
    println!("HsaEvtId::Codeobj = {:?}", rocm_rs::rocprofiler::HsaEvtId::Codeobj);

    println!("\nProfiling completed successfully!");
    Ok(())
}