use rocm_rs::{
    hip::{kernel::AsKernelArg, *},
    kernel_args,
    rocm_kernel_macros::*,
};

const LEN: usize = 1024;

// initializing rust gpu kernel
amdgpu_kernel_init!();

// saxpy
// x = ax+y
#[amdgpu_global]
fn saxpy(a: u32, x_arr: *mut u32, y_arr: *const u32) {
    // retriving data from buffere by workitem
    let x = read_by_workgroup_id_x(x_arr);
    let y = read_by_workgroup_id_x(y_arr);

    // writing data back
    write_by_workitem_id_x(x_arr, a * x + y);
}

// compiling gpu kernel and embedding kernel code inside host executable
const KERNEL: &[u8] = include_bytes!(amdgpu_kernel_finalize!());

fn main() -> Result<()> {
    // setting up device
    let device = Device::new(0)?;
    device.set_current()?;

    // loading gpu kerenel (runs in runtime!)

    let module = Module::load_data(KERNEL)?;

    // acquiring function handle from gpu kernel
    let function = module.get_function("saxpy")?;

    // preparing host side buffers
    let mut x_host: Vec<u32> = vec![0; LEN];
    let mut y_host: Vec<u32> = vec![0; LEN];

    // x => 0,1,2...LEN
    // x => 0,2,4...LEN
    for i in 0..LEN {
        x_host[i] = i as u32;
        y_host[i] = (i * 2) as u32;
    }

    // preparing gpu side buffers
    let mut x = DeviceMemory::<u32>::new(LEN)?;
    let mut y = DeviceMemory::<u32>::new(LEN)?;

    x.copy_from_host(&x_host)?;
    y.copy_from_host(&y_host)?;
    let a = 10;

    // providing arguments for kernel
    let kernel_args = kernel_args!(a, x, y);

    // setting up launch args
    let grid_dim = Dim3 { x: 2, y: 1, z: 1 };
    let block_dim = Dim3 {
        x: (LEN / 2) as u32,
        y: 1,
        z: 1,
    };

    function.launch(grid_dim, block_dim, 0, None, kernel_args)?;

    // retriving computed data
    x.copy_to_host(&mut x_host)?;

    println!("Output: {:?}", &x_host[..256]);

    Ok(())
}
