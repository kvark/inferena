//! Forced ResNet-shaped dW diagnostic, not an automatically selected kernel or benchmark result.
use meganeura::{CoopPolicy, Graph, Session, SessionOptions, compile::ShaderEntry};
use std::{sync::Arc, time::Instant};

#[path = "../src/graphics.rs"]
mod graphics;

fn data(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0 - 0.5) * scale
        })
        .collect()
}

fn main() {
    let cooperative = match std::env::args().nth(1).as_deref() {
        Some("cooperative") => true,
        Some("scalar") => false,
        _ => panic!("choose scalar or cooperative"),
    };
    let gpu =
        Arc::new(meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).unwrap());
    eprintln!("GPU: {}", gpu.device_information().device_name);
    let caps = gpu.capabilities().cooperative_matrix;
    assert_eq!(caps.f16_tile, 16);
    let (batch, ci, co, h, w) = (4u32, 256u32, 256u32, 14u32, 14u32);
    let x = data((batch * ci * h * w) as usize, 7, 1.0);
    let dy = data((batch * co * h * w) as usize, 37, 1e-12);
    let mut graph = Graph::new();
    let input = graph.input("x", &[x.len()]);
    let grad_output = graph.input("dy", &[dy.len()]);
    let output = graph.conv2d_grad_weight(grad_output, input, ci, h, w, co, 3, 3, 1, 1, 1);
    graph.set_outputs(vec![output]);
    let mut plan = meganeura::compile::compile(&graph);
    assert_eq!(plan.dispatches.len(), 1);
    let tile = if cooperative { 32 } else { 64 };
    let dispatch = &mut plan.dispatches[0];
    dispatch.shader = if cooperative {
        ShaderEntry::Conv2dGradWeightGemmSmall
    } else {
        ShaderEntry::Conv2dGradWeightGemm
    };
    dispatch.use_coop = cooperative;
    dispatch.tuned_coop =
        cooperative.then_some(meganeura::tune::MatmulTile::CooperativeScaledF16 {
            subgroup_size: caps.subgroup_size,
        });
    dispatch.workgroups = [(ci * 9).div_ceil(tile), co.div_ceil(tile), 1];
    let mut session = Session::with_context_opts(
        plan,
        Arc::clone(&gpu),
        SessionOptions {
            coop: if cooperative {
                CoopPolicy::Auto
            } else {
                CoopPolicy::Disabled
            },
            ..Default::default()
        },
    );
    assert_eq!(
        session.plan().dispatches[0].tuned_coop.is_some(),
        cooperative
    );
    session.set_input("x", &x);
    session.set_input("dy", &dy);
    session.step();
    session.wait();
    let actual = session.read_output((co * ci * 9) as usize);
    let mut worst_ratio = 0.0f64;
    for sample in 0..128usize {
        let i = sample * (actual.len() - 1) / 127;
        let oc = i / (ci as usize * 9);
        let ic = (i / 9) % ci as usize;
        let (kh, kw) = ((i % 9) / 3, i % 3);
        let mut expected = 0.0;
        for n in 0..batch as usize {
            for oh in 0..h as usize {
                for ow in 0..w as usize {
                    let (ih, iw) = (oh as i32 + kh as i32 - 1, ow as i32 + kw as i32 - 1);
                    if ih >= 0 && iw >= 0 && ih < h as i32 && iw < w as i32 {
                        let xi = ((n * ci as usize + ic) * h as usize + ih as usize) * w as usize
                            + iw as usize;
                        let yi = ((n * co as usize + oc) * h as usize + oh) * w as usize + ow;
                        expected += f64::from(x[xi]) * f64::from(dy[yi]);
                    }
                }
            }
        }
        let ratio = (f64::from(actual[i]) - expected).abs() / (1e-5 * (1e-12 + expected.abs()));
        worst_ratio = worst_ratio.max(ratio);
        assert!(
            ratio <= 1.0,
            "f64 check: output {i}, actual {}, expected {expected}, ratio {ratio}",
            actual[i]
        );
    }
    eprintln!("128 f64 spots passed; worst tolerance ratio: {worst_ratio}");
    for _ in 0..20 {
        session.step();
        session.wait();
    }
    graphics::start_phase("training");
    let mut times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        session.step();
        session.wait();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "cooperative={cooperative}, median_ms={}",
        times[times.len() / 2]
    );
}
