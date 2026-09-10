//! Opt-in Linux NGFX SDK 0.9.2 trigger; the CLI still bounds capture duration/submits.

pub fn start_phase(phase: &str) {
    let Ok(selected) = std::env::var("INFERENA_NGFX_PHASE") else {
        return;
    };
    assert!(matches!(
        selected.as_str(),
        "inference" | "latency" | "training"
    ));
    assert!(
        std::env::var_os("INFERENA_NSYS").is_some(),
        "capture requires NVTX markers"
    );
    if selected != phase {
        return;
    }
    #[cfg(target_os = "linux")]
    unsafe {
        linux::start();
    }
    #[cfg(not(target_os = "linux"))]
    panic!("the experiment's NGFX SDK trigger is Linux-only");
}

#[cfg(target_os = "linux")]
mod linux {
    use std::ffi::{CStr, c_char, c_int, c_void};

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    unsafe fn call<T>(library: *mut c_void, name: &CStr, params: &mut T) {
        let address = unsafe { dlsym(library, name.as_ptr()) };
        assert!(!address.is_null(), "missing NGFX SDK export {name:?}");
        let function = unsafe {
            std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut T) -> c_int>(address)
        };
        assert_eq!(unsafe { function(params) }, 0, "NGFX SDK call {name:?}");
    }

    pub(super) unsafe fn start() {
        // RTLD_NOW | RTLD_NOLOAD: only use an already injected target, never inject here.
        let library = unsafe { dlopen(c"libWarpVizTarget.so".as_ptr(), 2 | 4) };
        assert!(
            !library.is_null(),
            "launch through ngfx with --start-with-ngfx-sdk"
        );
        // Versioned layouts from NGFX_GPUTrace_{Vulkan,Common}_Types.h, SDK 0.9.2.
        let mut version = 4u32 | (1 << 16);
        unsafe {
            call(
                library,
                c"NGFX_GPUTrace_InitializeTraceActivityVulkan",
                &mut version,
            )
        };
        #[repr(C)]
        struct Wait {
            version: u32,
            status: c_int,
            timeout_ms: c_int,
        }
        let mut wait = Wait {
            version: 12 | (1 << 16),
            status: 1,
            timeout_ms: 5000,
        };
        unsafe { call(library, c"NGFX_GPUTrace_WaitForStatus", &mut wait) };
        unsafe { call(library, c"NGFX_GPUTrace_StartTraceVulkan", &mut version) };
        assert_eq!(unsafe { dlclose(library) }, 0);
        eprintln!("[meganeura] NGFX SDK trace started after phase warmup");
    }
}
