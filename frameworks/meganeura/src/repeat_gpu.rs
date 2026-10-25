//! Exact GPU comparison with an already CPU-qualified output/gradient snapshot.
use bg::ShaderData as _;
use blade_graphics as bg;
use meganeura::{Mode, Session, compile::BufferRef};
use std::{collections::BTreeMap, sync::Arc};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    count: u32,
    stride: u32,
    pad: [u32; 2],
}
#[derive(blade_macros::ShaderData)]
struct Data {
    actual: bg::BufferPiece,
    anchor: bg::BufferPiece,
    mismatch: bg::BufferPiece,
    params: Params,
}
const SOURCE: &str = r#"
struct Params { count: u32, stride: u32, pad: vec2<u32> }
var<storage> actual: array<u32>;
var<storage> anchor: array<u32>;
var<storage, read_write> mismatch: atomic<u32>;
var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var changed = false;
    for (var word = id.x; word < params.count; word += params.stride) {
        changed = changed || actual[word] != anchor[word];
    }
    if changed { atomicOr(&mismatch, 1u); }
}
"#;

#[derive(Clone, Copy)]
enum Source {
    Output(usize),
    Gradient(BufferRef),
}
impl Source {
    fn piece(self, session: &Session) -> Option<bg::BufferPiece> {
        match self {
            Self::Output(index) => session.output_buffer(index),
            Self::Gradient(parameter) => session.gradient_buffer(parameter),
        }
    }
}

pub struct Snapshot {
    gpu: Arc<bg::Context>,
    encoder: bg::CommandEncoder,
    pipeline: bg::ComputePipeline,
    _shader: bg::Shader,
    zero: bg::Buffer,
    flag: bg::Buffer,
    read: bg::Buffer,
    tensors: Vec<(Source, u32, bg::Buffer)>,
    pub bytes: u64,
}
impl Snapshot {
    pub fn capture(session: &Session, lengths: &[usize], mode: Mode) -> Option<Self> {
        let gpu = session.context().clone();
        let plan = session.plan();
        let mut tensors = BTreeMap::<u32, (usize, Source)>::new();
        if plan.output_buffers.len() != lengths.len() {
            return None;
        }
        for (index, (&buffer, &len)) in plan.output_buffers.iter().zip(lengths).enumerate() {
            tensors.insert(buffer.0, (len, Source::Output(index)));
        }
        if mode == Mode::Training {
            for &(parameter, gradient) in &plan.param_grad_pairs {
                let len = plan.param_types.get(&parameter)?.num_elements();
                tensors
                    .entry(gradient.0)
                    .and_modify(|(n, _)| *n = (*n).max(len))
                    .or_insert((len, Source::Gradient(parameter)));
            }
        }
        let mut bytes = 0u64;
        for (&buffer, &(len, source)) in &tensors {
            source.piece(session)?;
            if len == 0
                || u32::try_from(len).is_err()
                || len.checked_mul(4)? > plan.buffers[buffer as usize]
            {
                return None;
            }
            bytes = bytes.checked_add((len as u64).checked_mul(4)?)?;
        }
        if bytes == 0 || bytes > 1024 * 1024 * 1024 {
            return None;
        }
        let memory = gpu.memory_stats();
        if memory.budget.saturating_sub(memory.usage) < bytes.saturating_mul(2) {
            return None;
        }
        let shader = gpu
            .try_create_shader(bg::ShaderDesc {
                source: SOURCE,
                naga_module: None,
            })
            .ok()?;
        let pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
            name: "exact checked tensors",
            data_layouts: &[&Data::layout()],
            compute: shader.at("main"),
        });
        let mut encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
            name: "exact qualification repeat",
            buffer_count: 2,
            manual_barriers: false,
        });
        let zero = gpu.create_buffer(bg::BufferDesc {
            name: "repeat zero",
            size: 4,
            memory: bg::Memory::Upload,
        });
        unsafe {
            zero.data().cast::<u32>().write(0);
        }
        gpu.sync_buffer(zero.into(), 4, bg::BufferTarget::Data);
        let flag = gpu.create_buffer(bg::BufferDesc {
            name: "repeat mismatch",
            size: 4,
            memory: bg::Memory::Device,
        });
        let read = gpu.create_buffer(bg::BufferDesc {
            name: "repeat result",
            size: 4,
            memory: bg::Memory::Download,
        });
        let tensors: Vec<_> = tensors
            .into_iter()
            .map(|(_, (len, source))| {
                let anchor = gpu.create_buffer(bg::BufferDesc {
                    name: "CPU-qualified anchor",
                    size: len as u64 * 4,
                    memory: bg::Memory::Device,
                });
                (source, len as u32, anchor)
            })
            .collect();
        encoder.start();
        {
            let mut pass = encoder.transfer("copy qualified tensors");
            for &(source, words, anchor) in &tensors {
                pass.copy_buffer_to_buffer(
                    source.piece(session).unwrap(),
                    anchor.into(),
                    u64::from(words) * 4,
                );
            }
        }
        let sync = gpu.submit(&mut encoder);
        assert!(gpu.wait_for(&sync, u32::MAX).unwrap());
        Some(Self {
            gpu,
            encoder,
            pipeline,
            _shader: shader,
            zero,
            flag,
            read,
            tensors,
            bytes,
        })
    }
    pub fn matches(&mut self, session: &Session) -> bool {
        if !Arc::ptr_eq(&self.gpu, &session.context()) {
            return false;
        }
        let Some(actual): Option<Vec<_>> = self
            .tensors
            .iter()
            .map(|&(source, _, _)| source.piece(session))
            .collect()
        else {
            return false;
        };
        self.encoder.start();
        self.encoder
            .transfer("clear mismatch")
            .copy_buffer_to_buffer(self.zero.into(), self.flag.into(), 4);
        {
            let mut pass = self.encoder.compute("compare every checked word");
            for (&(_, count, anchor), actual) in self.tensors.iter().zip(actual) {
                let groups = count.div_ceil(256).min(65535);
                let mut pc = pass.with(&self.pipeline);
                pc.bind(
                    0,
                    &Data {
                        actual,
                        anchor: anchor.into(),
                        mismatch: self.flag.into(),
                        params: Params {
                            count,
                            stride: groups * 256,
                            pad: [0; 2],
                        },
                    },
                );
                pc.dispatch([groups, 1, 1]);
            }
        }
        self.encoder
            .transfer("read mismatch")
            .copy_buffer_to_buffer(self.flag.into(), self.read.into(), 4);
        let sync = self.gpu.submit(&mut self.encoder);
        assert!(self.gpu.wait_for(&sync, u32::MAX).unwrap());
        unsafe { self.read.data().cast::<u32>().read() == 0 }
    }
}
impl Drop for Snapshot {
    fn drop(&mut self) {
        self.gpu.destroy_command_encoder(&mut self.encoder);
        self.gpu.destroy_compute_pipeline(&mut self.pipeline);
        for (_, _, buffer) in self.tensors.drain(..) {
            self.gpu.destroy_buffer(buffer);
        }
        for buffer in [self.zero, self.flag, self.read] {
            self.gpu.destroy_buffer(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "GPU regression: exact output/gradient snapshots and large-array tails"]
    fn all_gradient_words_and_tail_changes_are_observed() {
        let device_id = std::env::var("MEGANEURA_DEVICE_ID")
            .ok()
            .map(|s| s.parse().unwrap());
        let gpu = Arc::new(
            meganeura::runtime::init_gpu_context_with(meganeura::GpuOptions {
                device_id,
                ..Default::default()
            })
            .unwrap(),
        );
        for count in [1, 257, 65537, 16_777_217] {
            let mut graph = meganeura::Graph::new();
            let x = graph.input("x", &[count]);
            let w = graph.parameter("w", &[count]);
            let y = graph.mul(x, w);
            let loss = graph.mean_all(y);
            graph.set_outputs(vec![loss]);
            let (mut session, _) = meganeura::train::build(
                &graph,
                meganeura::SessionConfig {
                    gpu: Some(gpu.clone()),
                    mode: Mode::Training,
                    runtime: meganeura::SessionOptions {
                        coop: meganeura::CoopPolicy::Disabled,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            );
            let mut x = vec![1.; count];
            session.set_parameter("w", &vec![0.; count]);
            session.set_input("x", &x);
            session.step();
            session.wait();
            let initial_loss = session.read_loss();
            assert_eq!(initial_loss, 0.);
            let mut gradient = vec![0.; count];
            session.read_param_grad("w", &mut gradient);
            assert!(gradient.iter().all(|&g| g == 1. / count as f32));
            drop(gradient);
            let mut snapshot = Snapshot::capture(&session, &[1], Mode::Training)
                .expect("GPU snapshot unavailable");
            assert!(snapshot.matches(&session));
            if count > 1 {
                x[0] = 0.;
            }
            x[count - 1] = 2.;
            session.set_input("x", &x);
            session.step();
            session.wait();
            assert_eq!(
                session.read_loss(),
                initial_loss,
                "gradient-only mutation changed the loss"
            );
            assert!(
                !snapshot.matches(&session),
                "missed the final gradient element at {count}"
            );
            x.fill(1.);
            session.set_input("x", &x);
            session.step();
            session.wait();
            assert!(snapshot.matches(&session), "exact restoration at {count}");
        }
    }
}
