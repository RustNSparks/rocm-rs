//! bindgen_rocm - ROCm/HIP kernel build automation
//!
//! TEAM-507: Full parity with bindgen_cuda
//! This module automates the ROCm kernel build process with the same API as bindgen_cuda

#![deny(missing_docs)]
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::os::unix::io::FromRawFd;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Error messages
#[derive(Debug)]
pub enum Error {
    /// ROCm installation not found
    RocmNotFound,
    /// GPU architecture could not be detected
    GpuArchNotFound,
    /// Invalid glob pattern
    InvalidGlob(String),
    /// Invalid path in glob results
    InvalidPath(std::path::PathBuf),
    /// Kernel path does not exist
    KernelPathNotFound(std::path::PathBuf),
    /// Watch path does not exist
    WatchPathNotFound(std::path::PathBuf),
    /// Include path has no filename
    IncludePathNoFilename(std::path::PathBuf),
    /// Failed to copy include header
    CopyIncludeHeaderFailed(std::path::PathBuf, std::io::Error),
    /// Include path is not valid UTF-8
    IncludePathNotUtf8(std::path::PathBuf),
    /// Kernel path has no filename
    KernelPathNoFilename(std::path::PathBuf),
    /// Failed to get file metadata
    MetadataFailed(std::path::PathBuf, std::io::Error),
    /// Failed to get modified time
    ModifiedTimeFailed(std::path::PathBuf, std::io::Error),
    /// Kernel path has no file stem
    KernelPathNoStem(std::path::PathBuf),
    /// Failed to create file
    CreateFileFailed(std::path::PathBuf, std::io::Error),
    /// Failed to execute hipify-perl
    HipifyPerlFailed(std::path::PathBuf, std::io::Error),
    /// hipify-perl exited with non-zero status
    HipifyPerlNonZeroExit(std::path::PathBuf, std::process::ExitStatus),
    /// Output filename is not valid UTF-8
    OutputFilenameNotUtf8(std::path::PathBuf),
    /// Failed to spawn hipcc
    HipccSpawnFailed(std::io::Error),
    /// Failed to wait for hipcc output
    HipccWaitFailed(std::io::Error),
    /// hipcc compilation failed
    HipccCompilationFailed {
        /// Kernel path that failed
        kernel_path: std::path::PathBuf,
        /// Command that was run
        command: String,
        /// stdout from hipcc
        stdout: String,
        /// stderr from hipcc
        stderr: String,
    },
    /// Invalid HSACO glob pattern
    InvalidHsacoGlob(String),
    /// Failed to write to file
    WriteFailed(std::path::PathBuf, std::io::Error),
    /// Invalid RAYON_NUM_THREADS value
    InvalidRayonThreads(String),
    /// OUT_DIR environment variable not set
    OutDirNotSet,
    /// rocminfo failed to execute
    RocminfoFailed(std::io::Error),
    /// rocminfo output is not valid UTF-8
    RocminfoOutputNotUtf8,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::RocmNotFound => write!(f, "Could not find ROCm in standard locations. Set it manually using Builder().rocm_root(...) or set ROCM_PATH, ROCM_ROOT, or HIP_PATH environment variable."),
            Error::GpuArchNotFound => write!(f, "Could not detect GPU architecture. Set ROCM_GPU_ARCH environment variable."),
            Error::InvalidGlob(pattern) => write!(f, "Invalid glob pattern: {}", pattern),
            Error::InvalidPath(path) => write!(f, "Invalid path in glob results: {}", path.display()),
            Error::KernelPathNotFound(path) => write!(f, "Kernel path does not exist: {}", path.display()),
            Error::WatchPathNotFound(path) => write!(f, "Watch path does not exist: {}", path.display()),
            Error::IncludePathNoFilename(path) => write!(f, "Include path has no filename: {}", path.display()),
            Error::CopyIncludeHeaderFailed(path, err) => write!(f, "Failed to copy include header {}: {}", path.display(), err),
            Error::IncludePathNotUtf8(path) => write!(f, "Include path is not valid UTF-8: {}", path.display()),
            Error::KernelPathNoFilename(path) => write!(f, "Kernel path has no filename: {}", path.display()),
            Error::MetadataFailed(path, err) => write!(f, "Failed to get metadata for {}: {}", path.display(), err),
            Error::ModifiedTimeFailed(path, err) => write!(f, "Failed to get modified time for {}: {}", path.display(), err),
            Error::KernelPathNoStem(path) => write!(f, "Kernel path has no file stem: {}", path.display()),
            Error::CreateFileFailed(path, err) => write!(f, "Failed to create file {}: {}", path.display(), err),
            Error::HipifyPerlFailed(path, err) => write!(f, "Failed to execute hipify-perl for {}: {}", path.display(), err),
            Error::HipifyPerlNonZeroExit(path, status) => write!(f, "hipify-perl failed for {} with status: {}", path.display(), status),
            Error::OutputFilenameNotUtf8(path) => write!(f, "Output filename is not valid UTF-8: {}", path.display()),
            Error::HipccSpawnFailed(err) => write!(f, "Failed to spawn hipcc. Ensure that you have ROCm installed and that `hipcc` is in your PATH: {}", err),
            Error::HipccWaitFailed(err) => write!(f, "Failed to wait for hipcc output: {}", err),
            Error::HipccCompilationFailed { kernel_path, command, stdout, stderr } => {
                write!(f, "hipcc error while compiling {:?}:\n\n# CLI {}\n\n# stdout\n{}\n\n# stderr\n{}", kernel_path, command, stdout, stderr)
            },
            Error::InvalidHsacoGlob(pattern) => write!(f, "Invalid HSACO glob pattern: {}", pattern),
            Error::WriteFailed(path, err) => write!(f, "Failed to write to {}: {}", path.display(), err),
            Error::InvalidRayonThreads(val) => write!(f, "RAYON_NUM_THREADS is not set to a valid integer: {}", val),
            Error::OutDirNotSet => write!(f, "Expected OUT_DIR environment variable to be present. Is this running within `build.rs`?"),
            Error::RocminfoFailed(err) => write!(f, "Failed to execute `rocminfo`. Ensure that you have ROCm installed and that `rocminfo` is in your PATH: {}", err),
            Error::RocminfoOutputNotUtf8 => write!(f, "rocminfo output is not valid UTF-8"),
        }
    }
}

impl std::error::Error for Error {}

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
            |s| usize::from_str(&s).unwrap_or_else(|_| {
                eprintln!("Warning: RAYON_NUM_THREADS is not set to a valid integer: {}", s);
                num_cpus::get_physical()
            }),
        );

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus)
            .build_global()
            .ok();

        let out_dir = std::env::var("OUT_DIR")
            .unwrap_or_else(|_| {
                eprintln!("Warning: OUT_DIR environment variable not set. Using current directory.");
                ".".to_string()
            })
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
            .filter_map(|p| p.ok())
            .collect(),
    )
}

fn default_include() -> Option<Vec<PathBuf>> {
    Some(
        glob::glob("src/**/*.h")
            .ok()?
            .filter_map(|p| p.ok())
            .collect(),
    )
}

impl Builder {
    /// Setup the kernel paths. All path must be set at once and be valid files.
    pub fn kernel_paths<P: Into<PathBuf>>(mut self, paths: Vec<P>) -> Result<Self, Error> {
        let paths: Vec<_> = paths.into_iter().map(|p| p.into()).collect();
        for path in &paths {
            if !path.exists() {
                return Err(Error::KernelPathNotFound(path.clone()));
            }
        }
        self.kernel_paths = paths;
        Ok(self)
    }

    /// Setup the paths that the lib depend on but does not need to build
    pub fn watch<T, P>(mut self, paths: T) -> Result<Self, Error>
    where
        T: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        let paths: Vec<_> = paths.into_iter().map(|p| p.into()).collect();
        for path in &paths {
            if !path.exists() {
                return Err(Error::WatchPathNotFound(path.clone()));
            }
        }
        self.watch = paths;
        Ok(self)
    }

    /// Setup the kernel paths. All path must be set at once and be valid files.
    pub fn include_paths<P: Into<PathBuf>>(mut self, paths: Vec<P>) -> Self {
        self.include_paths = paths.into_iter().map(|p| p.into()).collect();
        self
    }

    /// Setup the kernels with a glob.
    pub fn kernel_paths_glob(mut self, glob_pattern: &str) -> Result<Self, Error> {
        self.kernel_paths = glob::glob(glob_pattern)
            .map_err(|_| Error::InvalidGlob(glob_pattern.to_string()))?
            .map(|p| p.map_err(|e| Error::InvalidPath(e.path().to_path_buf())))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self)
    }

    /// Setup the include files with a glob.
    pub fn include_paths_glob(mut self, glob_pattern: &str) -> Result<Self, Error> {
        self.include_paths = glob::glob(glob_pattern)
            .map_err(|_| Error::InvalidGlob(glob_pattern.to_string()))?
            .map(|p| p.map_err(|e| Error::InvalidPath(e.path().to_path_buf())))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self)
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
        let rocm_root = self.rocm_root.ok_or(Error::RocmNotFound)?;
        let gpu_arch = self.gpu_arch.ok_or(Error::GpuArchNotFound)?;
        let rocm_include_dir = rocm_root.join("include");
        println!(
            "cargo:rustc-env=ROCM_INCLUDE_DIR={}",
            rocm_include_dir.display()
        );
        let out_dir = self.out_dir;

        let mut include_paths = self.include_paths;
        for path in &mut include_paths {
            println!("cargo:rerun-if-changed={}", path.display());
            let filename = path.file_name()
                .ok_or_else(|| Error::IncludePathNoFilename(path.clone()))?;
            let destination = out_dir.join(filename);
            std::fs::copy(path.clone(), &destination)
                .map_err(|e| Error::CopyIncludeHeaderFailed(path.clone(), e))?;
            path.pop();
        }

        include_paths.sort();
        include_paths.dedup();

        #[allow(unused)]
        let mut include_options: Vec<String> = include_paths
            .into_iter()
            .map(|s| {
                let path_str = s.clone().into_os_string()
                    .into_string()
                    .map_err(|_| Error::IncludePathNotUtf8(s.clone()))?;
                Ok("-I".to_string() + &path_str)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        include_options.push(format!("-I{}", rocm_include_dir.display()));

        println!("cargo:rerun-if-env-changed=HIPCC_FLAGS");
        for path in &self.watch {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        
        let children: Result<Vec<_>, Error> = self.kernel_paths
            .par_iter()
            .map(|p| -> Result<Option<(PathBuf, String, Result<std::process::Output, std::io::Error>)>, Error> {
                println!("cargo:rerun-if-changed={}", p.display());
                let mut output = p.clone();
                output.set_extension("hsaco");
                let filename = output.file_name()
                    .ok_or_else(|| Error::KernelPathNoFilename(p.clone()))?;
                let output_filename = std::path::Path::new(&out_dir)
                    .to_path_buf()
                    .join("out")
                    .with_file_name(filename);

                let ignore = if let Ok(metadata) = output_filename.metadata() {
                    let out_modified = metadata.modified()
                        .map_err(|e| Error::ModifiedTimeFailed(output_filename.clone(), e))?;
                    let in_metadata = p.metadata()
                        .map_err(|e| Error::MetadataFailed(p.clone(), e))?;
                    let in_modified = in_metadata.modified()
                        .map_err(|e| Error::ModifiedTimeFailed(p.clone(), e))?;
                    out_modified.duration_since(in_modified).is_ok()
                } else {
                    false
                };
                
                if ignore {
                    Ok(None)
                } else {
                    // Try hipify-perl first
                    let file_stem = p.file_stem()
                        .ok_or_else(|| Error::KernelPathNoStem(p.clone()))?;
                    let hip_file = out_dir.join(file_stem).with_extension("hip");
                    
                    // Try to run hipify-perl, but don't fail if it's not available
                    let hipify_result = std::process::Command::new("hipify-perl")
                        .arg(p)
                        .stdout(std::fs::File::create(&hip_file).unwrap_or_else(|_| {
                            std::fs::File::create("/dev/null").unwrap_or_else(|_| {
                                // Last resort: use stdout
                                unsafe { std::fs::File::from_raw_fd(1) }
                            })
                        }))
                        .status();
                    
                    // Only check status if hipify-perl ran successfully
                    if let Ok(status) = hipify_result {
                        if !status.success() {
                            return Err(Error::HipifyPerlNonZeroExit(p.clone(), status));
                        }
                    }
                    // If hipify-perl failed to run, just use the original file

                    let source = if hip_file.exists() { hip_file.clone() } else { p.clone() };
                    
                    let mut command = std::process::Command::new("hipcc");
                    command
                        .arg(format!("--offload-arch={}", gpu_arch))
                        .arg("-c")
                        .args(["-o", output_filename.to_str()
                            .ok_or_else(|| Error::OutputFilenameNotUtf8(output_filename.clone()))?])
                        .args(["-O3", "-ffast-math", "-fgpu-rdc"])
                        .args(&self.extra_args)
                        .args(&include_options);
                    
                    command.arg(&source);
                    let spawn_result = command.spawn()
                        .map_err(Error::HipccSpawnFailed)?;
                    Ok(Some((p.clone(), format!("{command:?}"), spawn_result.wait_with_output())))
                }
            })
            .collect();
        
        let children: Vec<_> = children?.into_iter().flatten().collect();

        let glob_pattern = format!("{0}/**/*.hsaco", out_dir.display());
        let hsaco_paths: Vec<PathBuf> = glob::glob(&glob_pattern)
            .map_err(|_| Error::InvalidHsacoGlob(glob_pattern.clone()))?
            .map(|p| p.map_err(|e| Error::InvalidPath(e.path().to_path_buf())))
            .collect::<Result<Vec<_>, _>>()?;
        
        let write = !children.is_empty() || self.kernel_paths.len() < hsaco_paths.len();
        
        for (kernel_path, command, child) in children {
            let output = child.map_err(Error::HipccWaitFailed)?;
            if !output.status.success() {
                return Err(Error::HipccCompilationFailed {
                    kernel_path,
                    command,
                    stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                });
            }
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
            let out_path = out.as_ref();
            let mut file = std::fs::File::create(out_path)
                .map_err(|e| Error::CreateFileFailed(out_path.to_path_buf(), e))?;
            for kernel_path in &self.paths {
                let file_stem = kernel_path
                    .file_stem()
                    .ok_or_else(|| Error::KernelPathNoStem(kernel_path.clone()))?;
                let name = file_stem
                    .to_str()
                    .ok_or_else(|| Error::IncludePathNotUtf8(kernel_path.clone()))?;
                file.write_all(
                    format!(
                        r#"pub const {}: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/{}.hsaco"));"#,
                        name.to_uppercase().replace('.', "_"),
                        name
                    )
                    .as_bytes(),
                )
                .map_err(|e| Error::WriteFailed(out_path.to_path_buf(), e))?;
                file.write_all(&[b'\n'])
                    .map_err(|e| Error::WriteFailed(out_path.to_path_buf(), e))?;
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
        .map_err(Error::RocminfoFailed)?;
    
    let out = std::str::from_utf8(&out.stdout)
        .map_err(|_| Error::RocminfoOutputNotUtf8)?;
    
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
