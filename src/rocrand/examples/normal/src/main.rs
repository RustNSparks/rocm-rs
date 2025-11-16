use rocm_rs::rocrand::utils::generate_normal_f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let size = 32;
    let mut host = vec![0f32; size];

    let device = generate_normal_f32(size, 0.5, 0.5, None)?;
    device.copy_to_host(&mut host)?;

    println!("{:?}", host);

    Ok(())
}
