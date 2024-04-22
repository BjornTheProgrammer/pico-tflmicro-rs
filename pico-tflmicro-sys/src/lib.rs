#![no_std]
mod bindings;

pub use bindings::root::*;
pub use cty;

fn hello() {
	// TFLITE_SCHEMA_VERSION
}

#[no_mangle]
extern "C" fn __wrap_main() -> i32 {
    return 0;
}
