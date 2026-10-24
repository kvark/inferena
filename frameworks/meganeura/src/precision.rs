//! Explicit arithmetic experiments within the accelerated profile.
//!
//! Auto preserves the collection path. Other policies are diagnostics: they
//! must qualify against an independent F32 reference before their timings can
//! justify a change to the collection's search space.
use meganeura::{CoopPolicy, Graph, SessionConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Auto,
    Fp16,
    Bf16Forward,
    Bf16,
    Bf16Inputs,
}

impl Policy {
    fn parse(value: &str, strict: bool) -> Result<Self, String> {
        let policy = match value {
            "auto" => Self::Auto,
            "fp16" => Self::Fp16,
            "bf16-forward" => Self::Bf16Forward,
            "bf16" => Self::Bf16,
            "bf16-inputs" => Self::Bf16Inputs,
            _ => {
                return Err(
                    "INFERENA_MEGANEURA_PRECISION must be auto, fp16, bf16-forward, bf16 or bf16-inputs".into(),
                );
            }
        };
        if strict && policy != Self::Auto {
            return Err("strict mode forbids reduced-precision overrides".into());
        }
        Ok(policy)
    }

    pub fn from_env() -> Self {
        Self::parse(
            &std::env::var("INFERENA_MEGANEURA_PRECISION").unwrap_or_else(|_| "auto".into()),
            strict(),
        )
        .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fp16 => "fp16",
            Self::Bf16Forward => "bf16-forward",
            Self::Bf16 => "bf16",
            Self::Bf16Inputs => "bf16-inputs",
        }
    }

    fn uses_bf16(self) -> bool {
        matches!(self, Self::Bf16Forward | Self::Bf16 | Self::Bf16Inputs)
    }

    pub fn graph(self, graph: &Graph) -> Option<Graph> {
        if self == Self::Bf16Inputs {
            return Some(bf16_inputs(graph));
        }
        self.uses_bf16().then(|| {
            graph
                .deep_clone()
                .into_bf16_matmuls()
                .into_bf16_convolutions()
        })
    }

    pub fn configure(self, config: &mut SessionConfig<'_>) {
        if self == Self::Fp16 {
            config.runtime.coop = CoopPolicy::AllowF16;
        }
        config.options.bf16_matrix_gradients = matches!(self, Self::Bf16 | Self::Bf16Inputs);
        if self.uses_bf16() {
            let caps = config
                .gpu
                .as_ref()
                .expect("owned GPU context")
                .capabilities();
            assert!(
                caps.shader_bfloat16
                    && meganeura::codegen::checked_cooperative_shape(
                        &caps.cooperative_matrix.bf16_f32_shapes
                    )
                    .is_some(),
                "BF16 experiment requires native BF16 storage and supported BF16/F32 cooperative matrices; use auto on this device"
            );
        }
    }

    pub fn metadata(self) -> serde_json::Value {
        let mut value = serde_json::json!({
            "comparison_class": if strict() { "strict-f32" } else { "accelerated" },
            "policy": self.name(),
            "diagnostic": self != Self::Auto,
            "cooperative_matrix_policy": if strict() { "NativeF32" } else if self == Self::Fp16 { "AllowF16" } else { "Auto" },
            "tensor_storage": "f32",
            "matmul_inputs": "forward: f16 for eligible cooperative-matrix kernels; f32 otherwise",
            "attention_inputs": "forward: f16 for eligible cooperative-matrix kernels; f32 otherwise",
            "convolution_inputs": "forward: f16 for eligible cooperative-matrix kernels; f32 otherwise",
            "backward_matmul_inputs": "f32",
            "backward_convolution_inputs": "f32",
            "backward_attention_inputs": "f32",
            "accumulation": "f32", "output": "f32", "loss": "f32",
            "reduced_precision_allowed": !strict(),
            "f16_cooperative_matrix_permitted": !strict(),
            "bf16_permitted": !strict(),
            "bf16_graph_conversion": self.uses_bf16(),
            "persistent_f16_tensors": false,
            "parameters": "f32; conversions, when needed, execute inside every timed step",
        });
        if strict() {
            for key in ["matmul_inputs", "attention_inputs", "convolution_inputs"] {
                value[key] = "f32".into();
            }
            value["native_f32_cooperative_matrix_permitted"] = true.into();
        } else if self == Self::Auto {
            value["tensor_storage"] =
                "f32 interfaces; private Winograd transforms may use f16/bf16".into();
            value["matmul_inputs"] =
                "qualified search over f32, f16 and bf16 multiplication; f32 storage/output".into();
            value["convolution_inputs"] =
                "qualified f32/f16/bf16 forward products and private Winograd transforms; f32 interfaces".into();
            value["matrix_search_candidates"] = serde_json::json!(["F16", "BF16"]);
        } else if self == Self::Fp16 {
            for key in ["backward_matmul_inputs", "backward_convolution_inputs"] {
                value[key] = "f16 for eligible cooperative kernels; f32 otherwise".into();
            }
        } else if self.uses_bf16() {
            value["tensor_storage"] = "mixed".into();
            value["storage_dtypes"] = serde_json::json!(["f32", "bf16"]);
            value["matmul_inputs"] =
                "bf16 ordinary dense products; other products follow Auto".into();
            value["convolution_inputs"] = "bf16 forward convolutions".into();
            value["matrix_output"] = if self == Self::Bf16Inputs {
                "f32"
            } else {
                "bf16 widened to f32 at interfaces"
            }
            .into();
            if matches!(self, Self::Bf16 | Self::Bf16Inputs) {
                value["backward_matmul_inputs"] =
                    "bf16 derivatives of explicit bf16 dense products; f32 otherwise".into();
            }
        }
        value
    }
}

/// Isolate operand rounding from the convenience transform's additional BF16
/// result rounding. Preserve original nodes and annotations, appending shared
/// operand casts, then restore topological order for compilation.
fn bf16_inputs(source: &Graph) -> Graph {
    use meganeura::graph::{DType, Op};
    let mut graph = source.deep_clone();
    let mut casts = std::collections::HashMap::new();
    for node in source.nodes() {
        if node.requires_full_precision
            || node.ty.dtype != DType::F32
            || !matches!(
                node.op,
                Op::MatMul
                    | Op::MatMulAT
                    | Op::MatMulBT
                    | Op::FusedMatMulAdd
                    | Op::FusedMatMulATAdd
                    | Op::FusedMatMulBTAdd
                    | Op::Conv2d { .. }
            )
            || node.inputs[..2]
                .iter()
                .any(|&id| source.node(id).ty.dtype != DType::F32)
        {
            continue;
        }
        for (index, &input) in node.inputs[..2].iter().enumerate() {
            let cast = *casts.entry(input).or_insert_with(|| graph.to_bf16(input));
            graph.nodes_mut()[node.id as usize].inputs[index] = cast;
        }
        graph.nodes_mut()[node.id as usize].matmul_impl = None;
    }
    graph.toposort()
}

pub fn strict() -> bool {
    std::env::var("INFERENA_STRICT").as_deref() == Ok("1")
}

/// Count products, not their conversion/transform dispatches. Winograd's
/// operands can be private BF16 tensors even while every graph interface is F32.
pub fn bf16_matrix_dispatches(plan: &meganeura::compile::ExecutionPlan) -> usize {
    use meganeura::{
        DType,
        compile::{Kernel, ShaderEntry},
    };
    plan.dispatches
        .iter()
        .filter(|d| {
            matches!(d.kernel,
            Kernel::CooperativeTyped { types, .. }
                | Kernel::CooperativeWorkgroup { types: Some(types), .. }
                if types.uses(DType::BF16))
                || matches!(
                    d.kernel,
                    Kernel::WinogradMatmul {
                        storage: DType::BF16,
                        ..
                    }
                )
                || matches!(d.shader, ShaderEntry::Conv2dBF16 { .. })
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_receipts_count_winograd_products_but_not_transforms() {
        use meganeura::{
            DType,
            codegen::MatrixTypes,
            compile::{Dispatch, Kernel, ShaderEntry},
        };
        let mut plan = meganeura::compile::compile(&Graph::new());
        for kernel in [
            Kernel::WinogradTransform(DType::BF16),
            Kernel::WinogradTransform(DType::F16),
            Kernel::WinogradMatmul {
                storage: DType::BF16,
                tile: [32, 32],
            },
            Kernel::WinogradMatmul {
                storage: DType::F16,
                tile: [32, 32],
            },
            Kernel::CooperativeTyped {
                types: MatrixTypes {
                    multiply: DType::BF16,
                    ..MatrixTypes::F32_AS_F16
                },
                splits: 1,
            },
            Kernel::CooperativeTyped {
                types: MatrixTypes::F32_AS_F16,
                splits: 1,
            },
        ] {
            plan.dispatches.push(Dispatch {
                kernel,
                ..Default::default()
            });
        }
        assert_eq!(bf16_matrix_dispatches(&plan), 2);
        plan.dispatches.push(Dispatch {
            shader: ShaderEntry::Conv2dBF16 {
                kernel_h: 3,
                kernel_w: 3,
                stride: 1,
                output_bf16: false,
            },
            ..Default::default()
        });
        assert_eq!(bf16_matrix_dispatches(&plan), 3);
    }

    #[test]
    fn strict_cannot_enable_a_reduced_override() {
        for value in ["fp16", "bf16-forward", "bf16", "bf16-inputs", "typo"] {
            assert!(Policy::parse(value, true).is_err());
        }
        assert_eq!(Policy::parse("auto", true).unwrap(), Policy::Auto);
        assert_eq!(Policy::parse("bf16", false).unwrap(), Policy::Bf16);
        assert!(Policy::parse("typo", false).is_err());
    }

    #[test]
    fn input_only_bf16_preserves_f32_outputs_parameters_and_protected_work() {
        use meganeura::graph::{DType, Op};
        let mut graph = Graph::new();
        let x = graph.input("x", &[16, 16]);
        let w = graph.parameter("w", &[16, 16]);
        let reduced = graph.matmul(x, w);
        let protected = graph.matmul(x, w);
        graph.nodes_mut()[protected as usize].requires_full_precision = true;
        graph.set_outputs(vec![reduced, protected]);
        let converted = bf16_inputs(&graph);
        assert!(
            converted
                .outputs()
                .iter()
                .all(|&id| converted.node(id).ty.dtype == DType::F32)
        );
        assert!(
            converted
                .nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::Input { .. } | Op::Parameter { .. }))
                .all(|n| n.ty.dtype == DType::F32)
        );
        for node in converted
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::MatMul))
        {
            let dtype = if node.requires_full_precision {
                DType::F32
            } else {
                DType::BF16
            };
            assert!(
                node.inputs
                    .iter()
                    .all(|&id| converted.node(id).ty.dtype == dtype)
            );
        }
        assert_eq!(
            converted
                .nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::ToBF16))
                .count(),
            2
        );
        assert!(graph.nodes().iter().all(|n| n.ty.dtype == DType::F32));
    }
}
