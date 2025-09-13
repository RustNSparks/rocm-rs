use rocm_rs::{hip::DeviceMemory, rocrand};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut generator = rocrand::default_generator()?;

    let buff_size = 32;
    let mut buff = DeviceMemory::new(buff_size)?;
    let mut host = vec![0f32; buff_size];

    generator.generate_normal(&mut buff, 0.5, 0.5)?;

    buff.copy_to_host(&mut host)?;
    println!("{:?}", host);

    Ok(())
}
