/// impl helper macro for rocblas functions
#[macro_export]
macro_rules! impl_rocblas_func {
    ($trait_name:ident, $fn_type:ident, {$( $t:ty => $func:path ),* $(,)?}) => {
        $(
            impl $trait_name for $t {
                fn func() -> $fn_type<Self> {
                    $func
                }
            }
        )*
    };
}
#[macro_export]
macro_rules! impl_rocblas_func_inner {
    ($func:expr, $($arg:expr),+ $(,)?) => {{
        let status = unsafe { $func($($arg),+) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }};
}
#[macro_export]
macro_rules! impl_rocblas_traits {
    (
        $trait_name:ident,
        $fn_type:ident,
        $ffi_map:tt,
        $method_name:ident,
        ($($arg:ident : $arg_ty:ty),+ $(,)?),
        ($($fn_arg:ty),+ $(,)?),
        ($($call_arg:expr),+ $(,)?)
    ) => {
        type $fn_type<T> = unsafe extern "C" fn($($fn_arg),+) -> u32;

        pub trait $trait_name {
            fn func() -> $fn_type<Self>;

            unsafe fn $method_name(
                $($arg: $arg_ty),+
            ) -> Result<()> {
                impl_rocblas_func_inner!(
                    Self::func(),
                    $($call_arg),+
                )
            }
        }

        impl_rocblas_func!($trait_name, $fn_type, $ffi_map);
    };
}