//! Opt-in CPU diagnosis; never use these instrumented samples as paper timings.

#[cfg(target_os = "linux")]
mod linux {
    use std::{ffi::c_long, time::Instant};

    #[repr(C)]
    struct Timespec {
        seconds: c_long,
        nanos: c_long,
    }

    unsafe extern "C" {
        fn clock_gettime(clock: i32, value: *mut Timespec) -> i32;
        fn sched_getcpu() -> i32;
    }

    fn thread_ns() -> u64 {
        let mut value = Timespec {
            seconds: 0,
            nanos: 0,
        };
        // Linux CLOCK_THREAD_CPUTIME_ID; includes this thread's user and kernel CPU time.
        assert_eq!(unsafe { clock_gettime(3, &mut value) }, 0);
        value.seconds as u64 * 1_000_000_000 + value.nanos as u64
    }

    fn frequency() -> (i32, Option<u64>) {
        let cpu = unsafe { sched_getcpu() };
        let khz = std::fs::read_to_string(format!(
            "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
        ))
        .ok()
        .and_then(|s| s.trim().parse().ok());
        (cpu, khz)
    }

    pub struct Trace {
        path: std::path::PathBuf,
        rows: Vec<serde_json::Value>,
    }

    pub struct Sample {
        frequency: (i32, Option<u64>),
        wall: Instant,
        cpu: u64,
        step: (u64, u64),
    }

    impl Trace {
        pub fn new(phase: &str) -> Option<Self> {
            let path = std::path::PathBuf::from(std::env::var_os("INFERENA_HOST_TRACE")?);
            std::fs::create_dir_all(&path).unwrap();
            Some(Self {
                path: path.join(format!("{phase}.json")),
                rows: Vec::new(),
            })
        }

        pub fn start(&self) -> Sample {
            let frequency = frequency();
            Sample {
                frequency,
                wall: Instant::now(),
                cpu: thread_ns(),
                step: (0, 0),
            }
        }

        pub fn finish(&mut self, stage: &str, sample: Sample) {
            let cpu = thread_ns() - sample.cpu;
            let wall = sample.wall.elapsed().as_nanos() as u64;
            self.rows.push(serde_json::json!({
                "stage": stage, "cpu_start": sample.frequency, "cpu_end": frequency(),
                "step_wall_ns": sample.step.0, "step_thread_cpu_ns": sample.step.1,
                "wait_wall_ns": wall - sample.step.0, "wait_thread_cpu_ns": cpu - sample.step.1,
            }));
        }
    }

    impl Sample {
        pub fn after_step(&mut self) {
            self.step = (
                self.wall.elapsed().as_nanos() as u64,
                thread_ns() - self.cpu,
            );
        }
    }

    impl Drop for Trace {
        fn drop(&mut self) {
            let file = std::fs::File::create_new(&self.path).unwrap();
            serde_json::to_writer(std::io::BufWriter::new(file), &self.rows).unwrap();
        }
    }
}

#[cfg(target_os = "linux")]
pub use linux::*;

#[cfg(not(target_os = "linux"))]
pub struct Trace;
#[cfg(not(target_os = "linux"))]
pub struct Sample;
#[cfg(not(target_os = "linux"))]
impl Trace {
    pub fn new(_: &str) -> Option<Self> {
        assert!(
            std::env::var_os("INFERENA_HOST_TRACE").is_none(),
            "host diagnosis is Linux-only"
        );
        None
    }
    pub fn start(&self) -> Sample {
        unreachable!()
    }
    pub fn finish(&mut self, _: &str, _: Sample) {
        unreachable!()
    }
}
#[cfg(not(target_os = "linux"))]
impl Sample {
    pub fn after_step(&mut self) {
        unreachable!()
    }
}
