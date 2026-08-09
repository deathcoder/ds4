use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command, description: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to start {description}: {error}"));
    if !status.success() {
        panic!("{description} failed with {status}");
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src/metal_shim.h");
    println!("cargo:rerun-if-changed=src/metal_shim.m");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let object = output.join("metal_shim.o");
    let archive = output.join("librust_star_metal.a");
    let source = manifest.join("src/metal_shim.m");
    let architecture = match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        Ok("x86_64") => "x86_64",
        Ok(other) => panic!("unsupported macOS architecture for Metal shim: {other}"),
        Err(error) => panic!("missing CARGO_CFG_TARGET_ARCH: {error}"),
    };

    let mut compile = Command::new("xcrun");
    compile
        .arg("--sdk")
        .arg("macosx")
        .arg("clang")
        .arg("-arch")
        .arg(architecture)
        .arg("-O3")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-fobjc-arc")
        .arg("-fmodules")
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&object);
    run(&mut compile, "Objective-C Metal shim compilation");

    let mut archive_command = Command::new("xcrun");
    archive_command
        .arg("ar")
        .arg("crs")
        .arg(&archive)
        .arg(&object);
    run(&mut archive_command, "Metal shim archive creation");

    print_link_directives(&output);
}

fn print_link_directives(output: &Path) {
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=rust_star_metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
}
