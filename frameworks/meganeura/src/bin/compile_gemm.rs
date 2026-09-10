//! Matched-domain compiler diagnostic; preparation excludes allocation/execution.

use blade_graphics::{self as gpu, ShaderData as _};
use meganeura::codegen::{self, ShaderGroup};
use std::time::Instant;

#[path = "../compilation.rs"]
mod compilation;

#[derive(blade_macros::ShaderData)]
struct MatmulData {
    matrix_a: gpu::BufferPiece,
    matrix_b: gpu::BufferPiece,
    matrix_c: gpu::BufferPiece,
    params: [u32; 4],
}

fn input(elements: usize, operand: u32, scale: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let mut bits = (index as u32).wrapping_add(0x9e37_79b9u32.wrapping_mul(operand + 1));
            bits ^= bits >> 16;
            bits = bits.wrapping_mul(0x85eb_ca6b);
            bits ^= bits >> 13;
            ((bits >> 8) as f32 / 16_777_216.0 - 0.5) * scale
        })
        .collect()
}

fn main() {
    env_logger::init();
    let _trace = compilation::init();
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let [m, n, k, tile] = args[..] else {
        panic!("usage: compile_gemm M N K TILE");
    };
    assert!(m > 0 && n > 0 && k > 0 && matches!(tile, 32 | 64));
    let gemv = std::env::var("INFERENA_GEMM_GEMV").as_deref() == Ok("1");
    assert!(!gemv || (m == 1 && n % 4 == 0));
    let sizes =
        [m.checked_mul(k), k.checked_mul(n), m.checked_mul(n)].map(|size| size.unwrap() as usize);
    assert!(sizes.iter().sum::<usize>() * 4 <= 256 * 1024 * 1024);
    let context_start = Instant::now();
    let context = meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).unwrap();
    let context_ns = context_start.elapsed().as_nanos();
    let compile = |tile, role| {
        let _span = tracing::info_span!("gemm_candidate", tile, role).entered();
        let prepare = Instant::now();
        let module = if gemv {
            codegen::generate_module(ShaderGroup::MatMulGemv)
        } else if tile == 32 {
            codegen::generate_module_small(ShaderGroup::MatMul)
        } else {
            codegen::generate_module(ShaderGroup::MatMul)
        };
        let source_bytes = module.source.len();
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &module.source,
            naga_module: Some(module.module),
        });
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "matched-f32-gemm",
            data_layouts: &[&MatmulData::layout()],
            compute: shader.at("main"),
        });
        (pipeline, source_bytes, prepare.elapsed().as_nanos())
    };
    let warmup_ns = if std::env::var("INFERENA_GEMM_WARM_COMPILER").as_deref() == Ok("1") {
        assert!(
            !gemv,
            "GEMV validation does not compare compiler warmup tiles"
        );
        let (mut warmup, _, elapsed) = compile(96 - tile, "warmup");
        context.destroy_compute_pipeline(&mut warmup);
        Some(elapsed)
    } else {
        None
    };
    let (mut pipeline, source_bytes, prepare_ns) = compile(tile, "measured");
    let buffers = sizes.map(|size| {
        context.create_buffer(gpu::BufferDesc {
            name: "gemm-input-or-output",
            size: (size * 4) as u64,
            memory: gpu::Memory::Shared,
        })
    });
    let download = context.create_buffer(gpu::BufferDesc {
        name: "gemm-readback",
        size: (sizes[2] * 4) as u64,
        memory: gpu::Memory::Download,
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "gemm-validation",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut validation = Vec::new();
    for scale in [1.0f32, 1.0e-12] {
        let inputs = [input(sizes[0], 0, scale), input(sizes[1], 1, 1.0)];
        for (data, buffer) in inputs.iter().zip(&buffers) {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.data().cast(), data.len());
            }
        }
        unsafe {
            std::slice::from_raw_parts_mut(buffers[2].data().cast::<f32>(), sizes[2])
                .fill(f32::NAN);
        }
        encoder.start();
        {
            let mut pass = encoder.compute("gemm");
            let mut bound = pass.with(&pipeline);
            bound.bind(
                0,
                &MatmulData {
                    matrix_a: buffers[0].at(0),
                    matrix_b: buffers[1].at(0),
                    matrix_c: buffers[2].at(0),
                    params: [m, n, k, 0],
                },
            );
            bound.dispatch(if gemv {
                [n / 4, 1, 1]
            } else {
                [n.div_ceil(tile), m.div_ceil(tile), 1]
            });
        }
        encoder.transfer("readback").copy_buffer_to_buffer(
            buffers[2].at(0),
            download.at(0),
            (sizes[2] * 4) as u64,
        );
        let done = context.submit(&mut encoder);
        assert!(context.wait_for(&done, !0).unwrap());
        let result = unsafe { std::slice::from_raw_parts(download.data().cast::<f32>(), sizes[2]) };
        let (mut failures, mut max_error) = (0usize, 0.0f64);
        for (index, &actual) in result.iter().enumerate() {
            let (row, col) = (index / n as usize, index % n as usize);
            let reference: f64 = (0..k as usize)
                .map(|inner| {
                    inputs[0][row * k as usize + inner] as f64
                        * inputs[1][inner * n as usize + col] as f64
                })
                .sum();
            let error = (reference - actual as f64).abs();
            max_error = max_error.max(error);
            failures += usize::from(
                !actual.is_finite() || error > scale as f64 * 1.0e-5 + reference.abs() * 2.0e-4,
            );
        }
        validation.push(serde_json::json!({"scale": scale, "failures": failures, "max_abs_error": max_error, "elements": result.len()}));
    }
    println!(
        "{}",
        serde_json::json!({
            "engine": "meganeura", "shape": [m, n, k], "tile": tile,
            "gemv": gemv, "gemv_threads": std::env::var("MEGANEURA_GEMV_THREADS").ok(),
            "gpu": context.device_information().device_name, "source_bytes": source_bytes,
            "context_ns": context_ns, "warmup_ns": warmup_ns, "prepare_ns": prepare_ns,
            "dimensions": "runtime uniforms", "validation": validation,
        })
    );
    context.destroy_command_encoder(&mut encoder);
    context.destroy_compute_pipeline(&mut pipeline);
    context.destroy_buffer(download);
    for buffer in buffers {
        context.destroy_buffer(buffer);
    }
    assert!(
        validation.iter().all(|row| row["failures"] == 0),
        "full-output f64 qualification failed"
    );
}
