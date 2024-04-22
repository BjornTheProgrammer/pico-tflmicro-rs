// Build procedure is based on esp-idf-sys: https://github.com/esp-rs/esp-idf-sys/blob/155299bde700905fd2ddb040d7a13fb73559ac68/build/native/cargo_driver.rs

use std::path::PathBuf;

use anyhow::{Context, Error};
use embuild::build::LinkArgsBuilder;
use embuild::cmake::file_api::codemodel::target::CompileGroup;
use embuild::cmake::file_api::codemodel::Language;
use embuild::cmake::file_api::{ObjKind, Query};
use embuild::cmake::Config;
use embuild::{bindgen, cargo};

#[allow(unused_macros)]
macro_rules! p {
    ($($tokens: tt)*) => {
        println!("cargo:warning={}", format!($($tokens)*))
    }
}

pub fn get_sysroot(target: &str) -> Result<String, anyhow::Error> {
    let mut sysroot = cc::Build::new();
    let sysroot = sysroot
        .target(target)
        .get_compiler()
        .to_command()
        .arg("--print-sysroot")
        .output()
        .context("Couldn't find target GCC executable.")
        .and_then(|output| {
            if output.status.success() {
                Ok(String::from_utf8(output.stdout)?)
            } else {
                panic!("Couldn't read output from GCC.")
            }
        })?;

    Ok(sysroot.trim().to_string())
}

fn generate_bindings(compile_group: &CompileGroup, compiler: &PathBuf, target: &str) {
    let cpp_headers = get_cpp_headers(compiler).expect("Couldn't find headers");

    let sysroot = get_sysroot(target).expect("Failed to get sysroot");
    let binder = bindgen::Factory::from_cmake(compile_group)
        .unwrap()
        .with_linker(compiler)
        .with_sysroot(sysroot);

    let mut binder = binder
        .builder()
        .unwrap()
        .clang_arg(format!("-I/submodules/pico-tflmicro/src/tensorflow"))
        .derive_eq(true)
        .use_core()
        .clang_arg("-xc++")
        .clang_arg("-std=c++11")
        .allowlist_recursively(true)
        .prepend_enum_name(false)
        .impl_debug(true)
        .layout_tests(false)
        .enable_cxx_namespaces()
        .derive_default(true)
        .size_t_is_usize(true)
        .ctypes_prefix("cty")
        // Types - blacklist
        .blocklist_type("std")
        .derive_partialeq(true)
        .derive_eq(true)
        .detect_include_paths(false);

    p!("cpp headers: {:?}", cpp_headers);
    for include in cpp_headers {
        binder = binder.clang_arg(format!(
            "-I{}",
            include.into_os_string().into_string().unwrap()
        ));
    }

    let wrapper_header = "submodules/pico-tflmicro/wrapped/src/wrapped.h";

    println!("cargo:rerun-if-changed={}", wrapper_header);

    let _ = binder
        .allowlist_file(wrapper_header)
        .header(wrapper_header)
        .generate()
        .unwrap()
        .write_to_file("src/bindings.rs");
}

pub fn get_compiler(target: &str) -> String {
    let mut compiler = cc::Build::new();
    let compiler = compiler.target(target).get_compiler();
    let compiler = compiler.path();

    compiler
        .to_str()
        .expect(&format!("Failed to find compiler for target '{}'", target))
        .to_string()
}

pub fn get_cpp_headers(compiler: &PathBuf) -> Result<Vec<PathBuf>, anyhow::Error> {
    let mut cpp_headers = cc::Build::new();
    cpp_headers
        .cpp(true)
        .no_default_flags(true)
        .compiler(compiler)
        .get_compiler()
        .to_command()
        .arg("-E")
        .arg("-Wp,-v")
        .arg("-xc++")
        .arg(".")
        .output()
        .context("Couldn't find target GCC executable.")
        .and_then(|output| {
            // We have to scrape the gcc console output to find where
            // the c++ headers are. If we only needed the c headers we
            // could use `--print-file-name=include` but that's not
            // possible.
            let gcc_out = String::from_utf8(output.stderr)?;

            // Scrape the search paths
            let search_start = gcc_out.find("search starts here").unwrap();
            let search_paths: Vec<PathBuf> = gcc_out[search_start..]
                .split('\n')
                .map(|p| PathBuf::from(p.trim()))
                .filter(|path| path.exists())
                .collect();

            Ok(search_paths)
        })
}

fn main() {
    let target = "thumbv6m-none-eabi";
    let cmake_build_dir = cargo::out_dir().join("build");

    p!("cmake build dir: {}", cmake_build_dir.display());

    // Set CMake to output API files https://cmake.org/cmake/help/git-stage/manual/cmake-file-api.7.html
    let query = Query::new(
        &cmake_build_dir,
        "cargo",
        &[ObjKind::Codemodel, ObjKind::Toolchains, ObjKind::Cache],
    )
    .unwrap();

    // Build C part
    let dst = Config::new("submodules/pico-tflmicro")
        .generator("Ninja")
        .build_target("exe")
        .build();

    p!("dst: {:?}", dst.display());
    println!("cargo:rustc-link-search={}/{}", cmake_build_dir.display(), "wrapped");
    println!("cargo:rustc-link-lib=static=pico-tflmicro-wrapped");

    // Retrieve information from CMake API files
    let replies = query.get_replies().unwrap();
    let codemodel = replies.get_codemodel().unwrap();
    let exe_target = codemodel
        .into_first_conf()
        .get_target("exe")
        .unwrap()
        .unwrap();
    let link = exe_target.link.unwrap();
    let compile_group = exe_target
        .compile_groups
        .get(0)
        .expect("Failed to get compile group");

    let link_args = LinkArgsBuilder::try_from(&link)
        .unwrap()
        .linker("arm-none-eabi-gcc")
        .working_directory(&cmake_build_dir)
        .build()
        .unwrap();
    p!("linker args {:?}", link_args);
    // link_args.output();

    let compiler = replies.get_toolchains();

    let compiler = compiler
        .and_then(|mut t| {
            t.take(Language::C)
                .ok_or_else(|| Error::msg("No C toolchain"))
        })
        .and_then(|t| {
            t.compiler
                .path
                .ok_or_else(|| Error::msg("No compiler path set"))
        })
        .context("Could not determine the compiler from cmake")
        .unwrap();

    // #[cfg(feature = "generate_bindings")]
    generate_bindings(compile_group, &compiler, target);
}
