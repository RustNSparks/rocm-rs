//! bindgen_rocm - ROCm/HIP kernel build automation
//!
//! TEAM-507: Full parity with bindgen_cuda
//! This module automates the ROCm kernel build process with the same API as bindgen_cuda

#![deny(missing_docs)]
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Error messages
#[derive(Debug)]
pub enum Error {}

/// Core builder to setup the bindings options
#[derive(Debug)]
pub struct Builder {
    rocm_root: Option<PathBuf>,
    kernel_paths: Vec<PathBuf>,
    watch: Vec<PathBuf>,
    include_paths: Vec<PathBuf>,
    gpu_arch: Option<String>,
    out_dir: PathBuf,
    extra_args: Vec<&'static str>,
}

impl Default for Builder {
    fn default() -> Self {
        // Use only physical cores for rayon
        let num_cpus = std::env::var("RAYON_NUM_THREADS").map_or_else(
            |_| num_cpus::get_physical(),
            |s| usize::from_str(&s).expect("RAYON_NUM_THREADS is not set to a valid integer"),
        );

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus)
            .build_global()
            .ok();

        let out_dir = std::env::var("OUT_DIR")
            .expect("Expected OUT_DIR environment variable to be present, is this running within `build.rs`?")
            .into();

        let rocm_root = rocm_include_dir();
        let kernel_paths = default_kernels().unwrap_or_default();
        let include_paths = default_include().unwrap_or_default();
        let extra_args = vec![];
        let watch = vec![];
        let gpu_arch = gpu_arch().ok();
        
        Self {
            rocm_root,
            kernel_paths,
            watch,
            include_paths,
            extra_args,
            gpu_arch,
            out_dir,
        }
    }
}

/// Helper struct to create a rust file when building HSACO files
pub struct Bindings {
    write: bool,
    paths: Vec<PathBuf>,
}

fn default_kernels() -> Option<Vec<PathBuf>> {
    Some(
        glob::glob("src/**/*.hip")
            .ok()?
            .map(|p| p.expect("Invalid path"))
            .collect(),
    )
}

fn default_include() -> Option<Vec<PathBuf>> {
    Some(
        glob::glob("src/**/*.cuh")
            .ok()?
            .map(|p| p.expect("Invalid path"))
            .collect(),
    )
}

impl Builder {
    /// Setup the kernel paths. All path must be set at once and be valid files.
    pub fn kernel_paths<P: Into<PathBuf>>(mut self, paths: Vec<P>) -> Self {
        let paths: Vec<_> = paths.into_iter().map(|p| p.into()).collect();
        let inexistent_paths: Vec<_> = paths.iter().filter(|f| !f.exists()).collect();
        if !inexistent_paths.is_empty() {
            panic!("Kernels paths do not exist {inexistent_paths:?}");
        }
        self.kernel_paths = paths;
        self
    }

    /// Setup the paths that the lib depend on but does not need to build
    pub fn watch<T, P>(mut self, paths: T) -> Self
    where
        T: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        let paths: Vec<_> = paths.into_iter().map(|p| p.into()).collect();
        let inexistent_paths: Vec<_> = paths.iter().filter(|f| !f.exists()).collect();
        if !inexistent_paths.is_empty() {
            panic!("Kernels paths do not exist {inexistent_paths:?}");
        }
        self.watch = paths;
        self
    }

    /// Setup the kernel paths. All path must be set at once and be valid files.
    pub fn include_paths<P: Into<PathBuf>>(mut self, paths: Vec<P>) -> Self {
        self.include_paths = paths.into_iter().map(|p| p.into()).collect();
        self
    }

    /// Setup the kernels with a glob.
    pub fn kernel_paths_glob(mut self, glob: &str) -> Self {
        self.kernel_paths = glob::glob(glob)
            .expect("Invalid glob")
            .map(|p| p.expect("Invalid path"))
            .collect();
        self
    }

    /// Setup the include files with a glob.
    pub fn include_paths_glob(mut self, glob: &str) -> Self {
        self.include_paths = glob::glob(glob)
            .expect("Invalid glob")
            .map(|p| p.expect("Invalid path"))
            .collect();
        self
    }

    /// Modifies the output directory.
    pub fn out_dir<P: Into<PathBuf>>(mut self, out_dir: P) -> Self {
        self.out_dir = out_dir.into();
        self
    }

    /// Sets up extra hipcc compile arguments.
    pub fn arg(mut self, arg: &'static str) -> Self {
        self.extra_args.push(arg);
        self
    }

    /// Forces the ROCm root to a specific directory.
    pub fn rocm_root<P>(&mut self, path: P)
    where
        P: Into<PathBuf>,
    {
        self.rocm_root = Some(path.into());
    }

    /// Consumes the builder and outputs 1 hsaco file for each kernel found.
    /// This function returns [`Bindings`] which can then be used
    /// to create a rust source file that will include those kernels.
    pub fn build_hsaco(self) -> Result<Bindings, Error> {
        let rocm_root = self.rocm_root.expect("Could not find ROCm in standard locations, set it manually using Builder().rocm_root(...)");
        let gpu_arch = self.gpu_arch.expect("Could not find gpu_arch");
        let rocm_include_dir = rocm_root.join("include");
        println!(
            "cargo:rustc-env=ROCM_INCLUDE_DIR={}",
            rocm_include_dir.display()
        );
        let out_dir = self.out_dir;

        let mut include_paths = self.include_paths;
        for path in &mut include_paths {
            println!("cargo:rerun-if-changed={}", path.display());
            let destination =
                out_dir.join(path.file_name().expect("include path to have filename"));
            std::fs::copy(path.clone(), destination).expect("copy include headers");
            path.pop();
        }

        include_paths.sort();
        include_paths.dedup();

        #[allow(unused)]
        let mut include_options: Vec<String> = include_paths
            .into_iter()
            .map(|s| {
                "-I".to_string()
                    + &s.into_os_string()
                        .into_string()
                        .expect("include option to be valid string")
            })
            .collect::<Vec<_>>();
        include_options.push(format!("-I{}", rocm_include_dir.display()));

        println!("cargo:rerun-if-env-changed=HIPCC_FLAGS");
        for path in &self.watch {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        
        let children = self.kernel_paths
            .par_iter()
            .flat_map(|p| {
                println!("cargo:rerun-if-changed={}", p.display());
                let mut output = p.clone();
                output.set_extension("hsaco");
                let output_filename = std::path::Path::new(&out_dir)
                    .to_path_buf()
                    .join("out")
                    .with_file_name(output.file_name().expect("kernel to have a filename"));

                let ignore = if let Ok(metadata) = output_filename.metadata() {
                    let out_modified = metadata.modified().expect("modified to be accessible");
                    let in_modified = p.metadata().expect("input to have metadata").modified().expect("input metadata to be accessible");
                    out_modified.duration_since(in_modified).is_ok()
                } else {
                    false
                };
                
                if ignore {
                    None
                } else {
                    // Try hipify-perl first
                    let hip_file = out_dir.join(p.file_stem().unwrap()).with_extension("hip");
                    let _ = std::process::Command::new("hipify-perl")
                        .arg(p)
                        .stdout(std::fs::File::create(&hip_file).ok().unwrap_or_else(|| {
                            std::fs::File::create("/dev/null").unwrap()
                        }))
                        .status();

                    let source = if hip_file.exists() { &hip_file } else { p };
                    
                    let mut command = std::process::Command::new("hipcc");
                    command
                        .arg(format!("--offload-arch={}", gpu_arch))
                        .arg("-c")
                        .args(["-o", output_filename.to_str().expect("valid outfile")])
                        .args(["-O3", "-ffast-math", "-fgpu-rdc"])
                        .args(&self.extra_args)
                        .args(&include_options);
                    
                    command.arg(source);
                    Some((p, format!("{command:?}"), command.spawn()
                        .expect("hipcc failed to start. Ensure that you have ROCm installed and that `hipcc` is in your PATH.")
                        .wait_with_output()))
                }
            })
            .collect::<Vec<_>>();

        let hsaco_paths: Vec<PathBuf> = glob::glob(&format!("{0}/**/*.hsaco", out_dir.display()))
            .expect("valid glob")
            .map(|p| p.expect("valid path for HSACO"))
            .collect();
        
        let write = !children.is_empty() || self.kernel_paths.len() < hsaco_paths.len();
        
        for (kernel_path, command, child) in children {
            let output = child.expect("hipcc failed to run. Ensure that you have ROCm installed and that `hipcc` is in your PATH.");
            assert!(
                output.status.success(),
                "hipcc error while compiling {kernel_path:?}:\n\n# CLI {command} \n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        
        Ok(Bindings {
            write,
            paths: self.kernel_paths,
        })
    }
}

impl Bindings {
    /// Writes a helper rust file that will include the HSACO sources as
    /// `const KERNEL_NAME` making it easier to interact with the HSACO sources.
    pub fn write<P>(&self, out: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        if self.write {
            let mut file = std::fs::File::create(out.as_ref()).expect("Create lib");
            for kernel_path in &self.paths {
                let name = kernel_path
                    .file_stem()
                    .expect("kernel to have stem")
                    .to_str()
                    .expect("kernel path to be valid");
                file.write_all(
                    format!(
                        r#"pub const {}: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/{}.hsaco"));"#,
                        name.to_uppercase().replace('.', "_"),
                        name
                    )
                    .as_bytes(),
                )
                .expect("write to file");
                file.write_all(&[b'\n']).expect("write to file");
            }
        }
        Ok(())
    }
}

fn rocm_include_dir() -> Option<PathBuf> {
    let env_vars = [
        "ROCM_PATH",
        "ROCM_ROOT",
        "HIP_PATH",
    ];
    
    #[allow(unused)]
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .map(Into::<PathBuf>::into);

    let roots = [
        "/opt/rocm",
        "/usr",
        "/usr/local/rocm",
    ];

    println!("cargo:info={roots:?}");

    #[allow(unused)]
    let roots = roots.into_iter().map(Into::<PathBuf>::into);

    #[cfg(not(feature = "ci-check"))]
    env_vars
        .chain(roots)
        .find(|path| path.join("include").join("hip").join("hip_runtime.h").is_file())
}

fn gpu_arch() -> Result<String, Error> {
    println!("cargo:rerun-if-env-changed=ROCM_GPU_ARCH");

    // Try to parse GPU arch from env
    if let Ok(gpu_arch_str) = std::env::var("ROCM_GPU_ARCH") {
        println!("cargo:rustc-env=ROCM_GPU_ARCH={gpu_arch_str}");
        return Ok(gpu_arch_str);
    }

    // Use rocminfo to get the current GPU arch
    let out = std::process::Command::new("rocminfo")
        .output()
        .expect("`rocminfo` failed. Ensure that you have ROCm installed and that `rocminfo` is in your PATH.");
    
    let out = std::str::from_utf8(&out.stdout).expect("stdout is not a utf8 string");
    
    // Parse gfx architecture from rocminfo output
    for line in out.lines() {
        if line.trim().starts_with("Name:") && line.contains("gfx") {
            if let Some(gfx) = line.split_whitespace().find(|s| s.starts_with("gfx")) {
                let arch = gfx.to_string();
                println!("cargo:rustc-env=ROCM_GPU_ARCH={arch}");
                return Ok(arch);
            }
        }
    }

    // Fallback to common architectures
    let default_arch = "gfx1030".to_string(); // RDNA2
    println!("cargo:warning=Could not detect GPU arch, using default: {default_arch}");
    println!("cargo:rustc-env=ROCM_GPU_ARCH={default_arch}");
    Ok(default_arch)
}
