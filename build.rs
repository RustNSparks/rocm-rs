use std::env;
use std::path::PathBuf;
use std::fs;
use bindgen::CargoCallbacks;
use std::collections::HashSet;

// Define module configuration
struct ModuleConfig {
    name: String,
    lib_name: String,
    extra_includes: Vec<String>,
    extra_args: Vec<String>,
}

fn main() {
    // Path to ROCm installation
    if env::var("SKIP_BINDGEN").is_ok() {
        println!("cargo:warning=Skipping bindgen as SKIP_BINDGEN is set");
        return;
    }
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    println!("cargo:rustc-link-search={}/lib", rocm_path);

    // Configure all modules
    let modules = vec![
        ModuleConfig {
            name: "hip".to_string(),
            lib_name: "amdhip64".to_string(),
            extra_includes: vec![],
            extra_args: vec![],
        },
        ModuleConfig {
            name: "rocblas".to_string(),
            lib_name: "rocblas".to_string(),
            extra_includes: vec![],
            extra_args: vec![],
        },
        ModuleConfig {
            name: "rocsolver".to_string(),
            lib_name: "rocsolver".to_string(),
            extra_includes: vec![],
            extra_args: vec![],
        },
        ModuleConfig {
            name: "rocfft".to_string(),
            lib_name: "rocfft".to_string(),
            extra_includes: vec![],
            extra_args: vec![],
        },
        ModuleConfig {
            name: "rocsparse".to_string(),
            lib_name: "rocsparse".to_string(),
            extra_includes: vec![format!("{}/include/rocsparse/internal", rocm_path)],
            extra_args: vec![],
        },
        ModuleConfig {
            name: "miopen".to_string(),
            lib_name: "MIOpen".to_string(),
            extra_includes: vec![],
            extra_args: vec![],
        },
        ModuleConfig {
            name: "rocrand".to_string(),
            lib_name: "rocrand".to_string(),
            extra_includes: vec![],
            extra_args: vec![],
        },
    ];

    // Process each module
    let mut first_module = true;
    for module in modules {
        let preserve_fp_constants = first_module;
        first_module = false;
        generate_bindings(&module, &rocm_path, preserve_fp_constants);
    }

    // Print success message
    println!("cargo:warning=ROCm bindings generated successfully");
}

fn generate_bindings(module: &ModuleConfig, rocm_path: &str, preserve_fp_constants: bool) {
    // Link to the appropriate library
    println!("cargo:rustc-link-lib={}", module.lib_name);

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=include/{}.h", module.name);

    // Base clang args that all modules need
    let mut clang_args = vec![
        "-D__HIP_PLATFORM_AMD__".to_string(),
        format!("-I{}/include", rocm_path),
        "-x".to_string(), "c++".to_string(),
    ];

    // Only add stdint.h and stddef.h for modules that explicitly need them
    if vec!["miopen", "rocsparse"].contains(&module.name.as_str()) {
        clang_args.push("--include".to_string());
        clang_args.push("stdint.h".to_string());
        clang_args.push("--include".to_string());
        clang_args.push("stddef.h".to_string());
    }

    // Add module-specific includes
    for include in &module.extra_includes {
        clang_args.push(format!("-I{}", include));
    }

    // Add module-specific args
    for arg in &module.extra_args {
        clang_args.push(arg.clone());
    }

    // Build bindgen command
    let mut builder = bindgen::Builder::default()
        .header(format!("include/{}.h", module.name))
        // Block GNU C++ template stuff
        .blocklist_item("__gnu_cxx::__max")
        .blocklist_item("__gnu_cxx::__min")
        .blocklist_item("__gnu_cxx::.*")
        .blocklist_item("_Value")
        .opaque_type("_Value");

    // Only keep floating point constants in the first module
    if !preserve_fp_constants {
        // Block math.h/fenv.h floating point constants that are duplicated
        builder = builder
            .blocklist_item("FP_INT_UPWARD")
            .blocklist_item("FP_INT_DOWNWARD")
            .blocklist_item("FP_INT_TOWARDZERO")
            .blocklist_item("FP_INT_TONEARESTFROMZERO")
            .blocklist_item("FP_INT_TONEAREST")
            .blocklist_item("FP_NAN")
            .blocklist_item("FP_INFINITE")
            .blocklist_item("FP_ZERO")
            .blocklist_item("FP_SUBNORMAL")
            .blocklist_item("FP_NORMAL");
    }

    // Add common blocklist items for system headers
    builder = builder
        .blocklist_item("_GLIBCXX_.*")
        .blocklist_item("_FEATURES_H")
        .blocklist_item("__GLIBC.*")
        .blocklist_item("__USE_.*")
        .blocklist_item("_STDC_PREDEF_H")
        .blocklist_item("__STDC_.*");

    // Add all clang args
    for arg in &clang_args {
        builder = builder.clang_arg(arg);
    }

    // Generate bindings
    let bindings = builder
        .parse_callbacks(Box::new(CargoCallbacks::new()))
        .layout_tests(false) // Disable layout tests for faster compilation
        .generate()
        .unwrap_or_else(|e| {
            panic!("Unable to generate bindings for {}: {:?}", module.name, e);
        });

    // Create output directory
    let out_dir = PathBuf::from("src").join(&module.name);
    fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("Couldn't create directory for {}: {:?}", module.name, e));

    // Write the bindings
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .unwrap_or_else(|e| panic!("Couldn't write bindings for {}: {:?}", module.name, e));

    // Create a mod.rs file
    let mod_content = format!(
        "//! Bindings for {}\n//! Auto-generated - do not modify\n\npub mod bindings;\n\n// Re-export commonly used types\npub use bindings::*;\n",
        module.name
    );

    if preserve_fp_constants {
        println!("cargo:warning=Generated bindings for {} with FP constants preserved", module.name);
    } else {
        println!("cargo:warning=Generated bindings for {} with FP constants blocked", module.name);
    }
}