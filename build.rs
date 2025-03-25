use std::env;
use std::path::PathBuf;
use std::fs;
use bindgen::CargoCallbacks;

// Define module configuration with enhanced options
struct ModuleConfig {
    name: String,
    lib_name: String,
    extra_includes: Vec<String>,
    extra_args: Vec<String>,
    allowlist_prefixes: Vec<String>,  // Prefixes to allow (e.g., "rocblas_", "hip")
    dependencies: Vec<String>,         // Other modules this one depends on
    needs_stddef_stdint: bool,         // Whether this module needs stddef.h and stdint.h
    needs_cpp: bool,                   // Whether this module needs C++ support
}

fn main() {
    // Skip bindgen if requested
    if env::var("SKIP_BINDGEN").is_ok() {
        println!("cargo:warning=Skipping bindgen as SKIP_BINDGEN is set");
        return;
    }

    // Path to ROCm installation
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    println!("cargo:rustc-link-search={}/lib", rocm_path);

    // Configure all modules with detailed options
    let modules = vec![
        ModuleConfig {
            name: "hip".to_string(),
            lib_name: "amdhip64".to_string(),
            extra_includes: vec![],
            extra_args: vec![],
            allowlist_prefixes: vec![
                "hip".to_string(),
                "HIP".to_string(),
                "cuda".to_string()
            ],
            dependencies: vec![],
            needs_stddef_stdint: false,
            needs_cpp: true,
        },
        ModuleConfig {
            name: "rocblas".to_string(),
            lib_name: "rocblas".to_string(),
            extra_includes: vec![],
            extra_args: vec![],
            allowlist_prefixes: vec!["rocblas_".to_string()],
            dependencies: vec!["hip".to_string()],
            needs_stddef_stdint: false,
            needs_cpp: true,
        },
        ModuleConfig {
            name: "rocsolver".to_string(),
            lib_name: "rocsolver".to_string(),
            extra_includes: vec![],
            extra_args: vec![],
            allowlist_prefixes: vec!["rocsolver_".to_string()],
            dependencies: vec!["hip".to_string(), "rocblas".to_string()],
            needs_stddef_stdint: false,
            needs_cpp: true,
        },
        ModuleConfig {
            name: "rocfft".to_string(),
            lib_name: "rocfft".to_string(),
            extra_includes: vec![],
            extra_args: vec![],
            allowlist_prefixes: vec!["rocfft_".to_string()],
            dependencies: vec!["hip".to_string()],
            needs_stddef_stdint: false,
            needs_cpp: true,
        },
        ModuleConfig {
            name: "rocsparse".to_string(),
            lib_name: "rocsparse".to_string(),
            extra_includes: vec![format!("{}/include/rocsparse/internal", rocm_path)],
            extra_args: vec![],
            allowlist_prefixes: vec!["rocsparse_".to_string()],
            dependencies: vec!["hip".to_string()],
            needs_stddef_stdint: true,
            needs_cpp: true,
        },
        ModuleConfig {
            name: "miopen".to_string(),
            lib_name: "MIOpen".to_string(),
            extra_includes: vec![],
            extra_args: vec![],
            allowlist_prefixes: vec!["miopen".to_string(), "MIOPEN".to_string()],
            dependencies: vec!["hip".to_string()],
            needs_stddef_stdint: true,
            needs_cpp: true,
        },
        ModuleConfig {
            name: "rocrand".to_string(),
            lib_name: "rocrand".to_string(),
            extra_includes: vec![],
            extra_args: vec![],
            allowlist_prefixes: vec!["rocrand_".to_string()],
            dependencies: vec!["hip".to_string()],
            needs_stddef_stdint: false,
            needs_cpp: true,
        },
    ];

    // Sort modules by dependency order
    let sorted_modules = sort_modules_by_dependencies(&modules);

    // Process each module
    let mut first_module = true;
    for module_name in sorted_modules {
        let module = modules.iter().find(|m| m.name == module_name).unwrap();
        let preserve_fp_constants = first_module;
        first_module = false;
        generate_bindings(module, &rocm_path, preserve_fp_constants);
    }

    // Generate module imports for dependencies
    generate_mod_imports(&modules);

    // Print success message
    println!("cargo:warning=ROCm bindings generated successfully");
}

// Sort modules so dependencies are processed first
fn sort_modules_by_dependencies(modules: &[ModuleConfig]) -> Vec<String> {
    let mut result = Vec::new();
    let mut visited = std::collections::HashSet::new();

    // Recursive function to add a module and its dependencies
    fn visit(
        module_name: &str,
        modules: &[ModuleConfig],
        result: &mut Vec<String>,
        visited: &mut std::collections::HashSet<String>,
        visiting: &mut std::collections::HashSet<String>,
    ) {
        if visited.contains(module_name) {
            return;
        }

        if visiting.contains(module_name) {
            panic!("Circular dependency detected with module {}", module_name);
        }

        visiting.insert(module_name.to_string());

        // Find the module definition
        if let Some(module) = modules.iter().find(|m| m.name == module_name) {
            // Visit all dependencies first
            for dep in &module.dependencies {
                visit(dep, modules, result, visited, visiting);
            }

            // Now add this module
            result.push(module_name.to_string());
            visited.insert(module_name.to_string());
        }

        visiting.remove(module_name);
    }

    // Process all modules
    let mut visiting = std::collections::HashSet::new();
    for module in modules {
        visit(&module.name, modules, &mut result, &mut visited, &mut visiting);
    }

    result
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
    ];

    // Add C++ support if needed
    if module.needs_cpp {
        clang_args.push("-x".to_string());
        clang_args.push("c++".to_string());
    }

    // Only add stdint.h and stddef.h for modules that explicitly need them
    if module.needs_stddef_stdint {
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
        // Block standard headers to avoid too much inclusion
        .blocklist_file(".*stdlib.h")
        .blocklist_file(".*stdio.h")
        .blocklist_file("stdint.h")
        .blocklist_file("stddef.h")
        // Block GNU C++ template stuff
        .blocklist_item("__gnu_cxx::__max")
        .blocklist_item("__gnu_cxx::__min")
        .blocklist_item("__gnu_cxx::.*")
        .blocklist_item("_Value")
        .opaque_type("_Value");

    // Add allowlist prefixes if specified
    if !module.allowlist_prefixes.is_empty() {
        for prefix in &module.allowlist_prefixes {
            builder = builder
                .allowlist_function(&format!("{}.*", prefix))
                .allowlist_type(&format!("{}.*", prefix))
                .allowlist_var(&format!("{}.*", prefix));
        }
    }

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

    println!("cargo:warning=Generated bindings for {}", module.name);
}

// Generate mod.rs files with proper imports for dependencies
fn generate_mod_imports(modules: &[ModuleConfig]) {
    for module in modules {
        let out_dir = PathBuf::from("src").join(&module.name);

        // Basic content for all mod.rs files
        let mut mod_content = format!(
            "//! Bindings for {}\n//! Auto-generated - do not modify\n\n\
             pub mod bindings;\n\n\
             // Re-export all bindings\n\
             pub use bindings::*;\n",
            module.name
        );

        // Add imports for dependencies
        if !module.dependencies.is_empty() {
            mod_content.push_str("\n// Import dependencies\n");
            for dep in &module.dependencies {
                mod_content.push_str(&format!("pub use crate::{}::*;\n", dep));
            }
        }

        // Write the mod.rs file
        fs::write(out_dir.join("mod.rs"), mod_content)
            .unwrap_or_else(|e| panic!("Couldn't write mod.rs for {}: {:?}", module.name, e));
    }
}