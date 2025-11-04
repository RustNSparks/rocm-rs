use rocm_rs::hip::{self, DeviceMemory, memory::memory_ext::MemoryExt};

fn main() -> Result<(), hip::Error> {
    let arr: Vec<i32> = vec![
        87, 23, 56, 12, 91, 45, 78, 34, 67, 5, 99, 31, 64, 29, 76, 18, 50, 82, 37, 93, 15, 41, 60,
        27, 72, 11, 48, 80, 33, 66, 22, 55, 77, 10, 44, 88, 3, 39, 70, 25, 58, 9, 43, 75, 20, 53,
        85, 30, 63, 17, 51, 84, 28, 61, 14, 47, 79, 2, 35, 68, 19, 52, 81, 26, 59, 92, 13, 46, 71,
        24, 57, 90, 32, 65, 8, 40, 73, 16, 49, 83, 36, 69, 1, 38, 74, 21, 54, 86, 4, 42, 7, 62, 95,
        31, 64, 98, 12, 45, 78, 0,
    ];

    let mut host_sorted = arr.clone();
    host_sorted.sort();

    let mut device_arr = DeviceMemory::new(arr.len())?;

    device_arr.copy_from_host(&arr)?;

    device_arr.sort()?;

    let mut gpu_sroted_ascending = vec![0; arr.len()];
    device_arr.copy_to_host(&mut gpu_sroted_ascending)?;

    assert_eq!(host_sorted, gpu_sroted_ascending);
    println!("Sorted ascending: {:?}", gpu_sroted_ascending);

    host_sorted.reverse();

    device_arr.copy_from_host(&arr)?;

    device_arr.sort_desc()?;

    let mut gpu_sroted_descending = vec![0; arr.len()];
    device_arr.copy_to_host(&mut gpu_sroted_descending)?;

    assert_eq!(host_sorted, gpu_sroted_descending);
    println!("Sorted descending: {:?}", gpu_sroted_descending);

    Ok(())
}
