use std::path::PathBuf;

use rocm_kernel_macros::{amdgpu_kernel_attr, amdgpu_kernel_finalize, amdgpu_kernel_init};
use rocm_rs::hip::*;

const LEN: usize = 1024;

// initializing rust gpu kernel
amdgpu_kernel_init!();

// marking code that will be coppied to gpu kernel
#[amdgpu_kernel_attr]
fn kernel(input: *const u32, output: *mut u32) {
    // retriving data from buffere by workitem
    let num = read_by_workitem_id_x(input);

    // writing data back
    write_by_workitem_id_x(output, num * 3);
}

// compiling gpu kernel
const AMDGPU_KERNEL_BINARY_PATH: &str = amdgpu_kernel_finalize!();

fn main() -> Result<()> {
    // setting up device
    let device = Device::new(0)?;
    device.set_current()?;

    // Create a stream for async operations
    let stream = Stream::new()?;

    // loading gpu kerenel (runs in runtime!)
    let kernel_path = PathBuf::from(AMDGPU_KERNEL_BINARY_PATH);
    assert!(kernel_path.exists());

    let module = Module::load(kernel_path)?;

    // acquiring function handle from gpu kernel
    let function = module.get_function("kernel")?;

    // preparing host side buffers
    let mut in_host: Vec<u32> = vec![0; LEN];
    let out_host: Vec<u32> = vec![0; LEN];

    for i in 0..LEN {
        in_host[i] = i as u32;
    }

    // preparing gpu side buffers
    let input = DeviceMemory::<u32>::new(LEN)?;
    let output = DeviceMemory::<u32>::new(LEN)?;

    // Copy data from host to device
    input.copy_from_host_async(in_host, &stream)?;

    // providing arguments for kernel
    let kernel_args = [input.as_kernel_arg(), output.as_kernel_arg()];

    // setting up launch args
    let grid_dim = Dim3 { x: 2, y: 1, z: 1 };
    let block_dim = Dim3 {
        x: (LEN / 2) as u32,
        y: 1,
        z: 1,
    };

    function.launch(grid_dim, block_dim, 0, Some(&stream), &mut kernel_args.clone())?;

    // retriving computed data
    let pending = output.copy_to_host_async(out_host, &stream)?;

    // synchronizing memory (awaiting for copy to finish)
    let out_host = stream.synchronize_memory(pending)?;
    println!("Output: {:?}", &out_host[..256]);

    Ok(())
}
