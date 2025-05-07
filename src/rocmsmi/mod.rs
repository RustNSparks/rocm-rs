pub use rocm_smi_lib as rocmsmi;
pub use rocmsmi::*;

#[cfg(test)]
mod test {
    use crate::rocmsmi::{RocmSmi, *};

    #[test]
    fn rocm_smi_test() -> Result<(), rocmsmi::RocmErr> {
        let mut rocm_smi = RocmSmi::init()?;

        let _ = rocm_smi.get_device_identifiers(0).unwrap();
        Ok(())
    }
}
