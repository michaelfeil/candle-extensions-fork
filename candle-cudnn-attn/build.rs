use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Find cuDNN installation
    let cudnn_include = find_cudnn_include();
    let cudnn_lib = find_cudnn_lib();

    if cudnn_include.is_none() || cudnn_lib.is_none() {
        println!("cargo:warning=cuDNN not found, using stub implementation");
        generate_stub_bindings();
        return;
    }

    let cudnn_include = cudnn_include.unwrap();
    let cudnn_lib = cudnn_lib.unwrap();

    println!("cargo:rustc-link-search=native={}", cudnn_lib.display());
    println!("cargo:rustc-link-lib=dylib=cudnn");
    println!("cargo:rustc-link-lib=dylib=cudnn_ops");
    println!("cargo:rustc-link-lib=dylib=cudnn_adv");

    // Find CUDA include path
    let cuda_include = find_cuda_include();

    // Generate bindings using bindgen
    let mut builder = bindgen::Builder::default()
        .header(cudnn_include.join("cudnn.h").to_str().unwrap())
        .header(cudnn_include.join("cudnn_ops.h").to_str().unwrap())
        .clang_arg(format!("-I{}", cudnn_include.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("cudnn.*")
        .allowlist_type("cudnn.*")
        .allowlist_var("CUDNN.*");

    // Add CUDA include path if found
    if let Some(cuda_inc) = cuda_include {
        builder = builder.clang_arg(format!("-I{}", cuda_inc.display()));
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn find_cuda_include() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(path) = env::var("CUDA_INCLUDE_DIR") {
        return Some(PathBuf::from(path));
    }

    // Common paths - check CUDA 12.x locations first
    let paths = [
        "/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "/usr/local/cuda-12.6/targets/x86_64-linux/include",
        "/usr/local/cuda-12.4/targets/x86_64-linux/include",
        "/usr/local/cuda-12.2/targets/x86_64-linux/include",
        "/usr/local/cuda-12.0/targets/x86_64-linux/include",
        "/usr/local/cuda/include",
        "/usr/include/cuda",
        "/usr/local/include/cuda",
    ];

    for path in &paths {
        let cuda_runtime = PathBuf::from(path).join("cuda_runtime_api.h");
        if cuda_runtime.exists() {
            return Some(PathBuf::from(path));
        }
    }

    None
}

fn find_cudnn_include() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(path) = env::var("CUDNN_INCLUDE_DIR") {
        return Some(PathBuf::from(path));
    }

    // Common paths
    let paths = [
        "/usr/local/cuda/include",
        "/usr/include",
        "/usr/local/include",
    ];

    for path in &paths {
        let cudnn_h = PathBuf::from(path).join("cudnn.h");
        if cudnn_h.exists() {
            return Some(PathBuf::from(path));
        }
    }

    // Check Python package location
    let nvidia_path = PathBuf::from("/usr/local/lib/python3.12/dist-packages/nvidia");
    let cudnn_include = nvidia_path.join("cudnn").join("include");
    if cudnn_include.join("cudnn.h").exists() {
        return Some(cudnn_include);
    }

    None
}

fn find_cudnn_lib() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(path) = env::var("CUDNN_LIB_DIR") {
        return Some(PathBuf::from(path));
    }

    // Common paths
    let paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/lib64",
        "/usr/local/lib",
    ];

    for path in &paths {
        let lib_path = PathBuf::from(path);
        if lib_path.join("libcudnn.so").exists() || lib_path.join("libcudnn.so.9").exists() {
            return Some(lib_path);
        }
    }

    // Check Python package location
    let nvidia_path = PathBuf::from("/usr/local/lib/python3.12/dist-packages/nvidia");
    let cudnn_lib = nvidia_path.join("cudnn").join("lib");
    if cudnn_lib.join("libcudnn.so").exists() || cudnn_lib.join("libcudnn.so.9").exists() {
        return Some(cudnn_lib);
    }

    None
}

fn generate_stub_bindings() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::write(
        out_path.join("bindings.rs"),
        r#"// Stub cuDNN bindings - cuDNN not found
use libc::{c_char, c_int, c_void, size_t};

pub const CUDNN_STATUS_SUCCESS: c_int = 0;
pub const CUDNN_STATUS_NOT_INITIALIZED: c_int = 1;

pub type cudnnStatus_t = c_int;
pub type cudnnHandle_t = *mut c_void;

#[repr(C)]
pub struct cudnnContext {
    _unused: [u8; 0],
}

extern "C" {
    pub fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
    pub fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;
    pub fn cudnnGetErrorString(status: cudnnStatus_t) -> *const c_char;
    pub fn cudnnGetVersion() -> usize;
}
"#,
    )
    .expect("Couldn't write stub bindings!");
}
