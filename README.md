# inferena

[![CI](https://github.com/kvark/inferena/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/inferena/actions/workflows/ci.yml)

Single-GPU ML systems benchmark for matched inference and training workloads.
PyTorch and Meganeura currently report the complete timing, precision,
environment, output, and gradient metadata used for strict cross-engine
validation. Other runners remain available while their reporting is brought
up to the same standard.

Inspired by [meganeura's bench/compare.sh](https://github.com/kvark/meganeura/tree/main/bench) pipeline.

## Frameworks

All Rust frameworks use bleeding-edge git dependencies pinned to specific
revisions in `Cargo.toml`. The harness reads these at build time — framework
links in the results tables always point to the exact revision tested.

| Framework | Language | GPU Backend |
|-----------|----------|-------------|
| [PyTorch](https://pytorch.org/) | Python | CUDA / ROCm / XPU / MPS |
| [Candle](https://github.com/huggingface/candle) | Rust | CUDA / Metal / CPU |
| [Burn](https://github.com/tracel-ai/burn) | Rust | wgpu (Vulkan / Metal / DX12) |
| [Luminal](https://github.com/luminal-ai/luminal) | Rust | CUDA / Metal |
| [Meganeura](https://github.com/kvark/meganeura) | Rust | blade (Vulkan / Metal) |
| [Inferi](https://github.com/dimforge/inferi)[^inferi] | Rust | wgpu (Vulkan / Metal) / CUDA |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C++ | CUDA / Metal / Vulkan / CPU |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Python/C++ | CUDA / TensorRT / DirectML / CPU |
| [JAX](https://github.com/jax-ml/jax) | Python | CUDA / TPU / CPU |
| [MAX](https://github.com/modular/modular) | Python | CUDA / CPU |
| [MLX](https://github.com/ml-explore/mlx) | Python | Metal (Apple Silicon only) |

[^inferi]: Opt-in. Inferi compiles its shaders through rust-gpu, and the
    build script installs that toolchain, prompting for consent when it is
    missing — which fails in a build script, since there is no TTY. Install
    it first, then opt in:
    `cargo install cargo-gpu --version 0.10.0-alpha.1 && cargo gpu install`,
    then set `INFERENA_ENABLE_INFERI=1`. Other frameworks are unaffected;
    `run.sh` builds only the crates the requested `--frameworks` need.

## Platform support

| Framework | Linux | macOS | Windows |
|-----------|:-----:|:-----:|:-------:|
| PyTorch | CUDA, ROCm, XPU, CPU | MPS, CPU | CUDA, CPU |
| ONNX Runtime | CUDA, TensorRT, CPU | CoreML, CPU | DirectML, CPU |
| JAX | CUDA, TPU, CPU | CPU | CPU |
| MAX | CUDA, CPU | CPU | — |
| Candle | CUDA, CPU | Metal, CPU | CPU |
| Burn | Vulkan, CPU | Metal, CPU | — |
| Luminal | CUDA | Metal | — |
| Meganeura | Vulkan | Metal | Vulkan |
| Inferi | CUDA, Vulkan | Metal | CUDA, Vulkan |
| llama.cpp | CUDA, Vulkan, CPU | Metal, CPU | CUDA, Vulkan, CPU |
| MLX | — | Metal | — |

Frameworks that can't run on a given platform are reported as `✗` in the results.

## Models

Each model has its own page with architecture details, benchmark caveats,
and results tables.

| Model | Type | Params | Results |
|-------|------|-------:|---------|
| [SmolLM2-135M](models/SmolLM2-135M.md) | Text LLM | 135M | [results](models/SmolLM2-135M.md#results) |
| [SmolVLA](models/SmolVLA.md) | Robotics Action Expert | 99.85M | [results](models/SmolVLA.md#results) |
| [Conditioned diffusion U-Net](models/StableDiffusion.md) | Conv + attention U-Net | 10.93M | [results](models/StableDiffusion.md#results) |
| [ResNet-50](models/ResNet-50.md) | Image Classification (CNN) | 25.53M | [results](models/ResNet-50.md#results) |
| [Whisper-tiny encoder](models/Whisper-tiny.md) | Speech Encoder | 8.21M | [results](models/Whisper-tiny.md#results) |

## What it measures

For each instrumented framework and model, Inferena records:

1. **Compile** — Graph/compiler specialization and executable preparation
   (seconds; model loading and parameter upload are reported or excluded
   separately).
2. **Inference** — A no-gradient full forward pass with fixed deterministic
   input (milliseconds). For decoder LLMs this is explicitly the 128-token
   prefill measurement.
3. **Latency** — A matched single-token or minimal-batch workload
   (milliseconds). The current LLM workload is a stateless one-token forward
   without a KV cache, so it is not reported as decode latency.
4. **Training** — Forward, loss, and backward together, without an optimizer
   update (milliseconds).

Each series uses configurable warmups followed by retained raw samples and
reports the median and interquartile range. Correctness requires agreement in
canonical output shape, loss, and a deterministic 256-value sample spanning
the output tensor. The audited runners must also report total and
per-parameter gradient norms. A close loss by itself is not enough to validate
a result, and inference and training validity are recorded independently.

Precision is part of every result:

- The default is the practical accelerated configuration. It keeps f32
  storage/output but permits documented reduced-input, f32-accumulate hardware
  paths. PyTorch may use TF32; Meganeura may use f16 cooperative-matrix
  inputs.
- `--strict` disables those paths for an f32 control run. PyTorch TF32 and
  Meganeura f16 cooperative-matrix paths are disabled.

The two configurations are separate comparison classes because TF32 and f16
are not numerically equivalent.

## Prerequisites

- **Rust** toolchain (for Candle, Burn, Luminal, Meganeura)
- **Python 3** (3.12 recommended — best pre-built GPU wheel coverage)
- **GPU drivers** for your hardware

### Ubuntu/Debian system packages

```bash
# Common (Vulkan for Burn, Meganeura)
sudo apt install vulkan-tools libvulkan-dev glslc libssl-dev pkg-config

# NVIDIA GPU:
sudo apt install nvidia-driver-595          # driver (version may vary)
sudo apt install nvidia-cuda-toolkit        # nvcc — needed for Candle, Luminal
# CUDA runtime libraries are provided by pip packages (nvidia-cublas-cu12, etc.)

# AMD GPU:
# sudo apt install rocm-dev libdrm-common librocm-core1
# Some ROCm-built shared libs look for amdgpu.ids under /opt/amdgpu (AMD's
# prefix), but Ubuntu installs it under /usr/share. Symlink once:
#   sudo mkdir -p /opt/amdgpu/share/libdrm
#   sudo ln -s /usr/share/libdrm/amdgpu.ids /opt/amdgpu/share/libdrm/amdgpu.ids
# JAX's ROCm plugin hard-links rocprofiler-sdk + roctracer. On Ubuntu 26.04
# (resolute) those aren't packaged yet — AMD's official apt repo ships them
# only for noble/24.04. Without them JAX falls back to CPU; to get GPU JAX,
# manually `dpkg -i` the `~24.04_amd64.deb` files from
# https://repo.radeon.com/rocm/apt/7.1/pool/main/r/ for rocprofiler-register,
# rocprofiler-sdk, roctracer, rocm-core (ROCm-compatible but unmanaged).

# Intel GPU (Arc, Data Center GPU, Xe iGPU):
# sudo apt install libze-intel-gpu1 libze1 intel-opencl-icd
# SYCL/oneMKL runtime libs ship as dependencies of the torch+xpu wheel.

# Enable torch.compile on CPU (Inductor needs Python headers + g++):
# sudo apt install python3-dev g++
# Without these, the pytorch runner falls back to eager mode.
```

Run `./run.sh --check` after setup to verify what's working and get
install hints for anything missing.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-nvidia.txt       # NVIDIA CUDA
# pip install -r requirements-amd.txt        # AMD ROCm
# pip install -r requirements-intel.txt      # Intel XPU (Arc / Xe iGPU)
# pip install -r requirements-apple.txt      # Apple Metal
# pip install -r requirements-cpu.txt        # CPU only
./run.sh                                     # practical defaults, all models/frameworks
./run.sh -m SmolLM2-135M                     # single model
./run.sh -m SmolLM2-135M -f pytorch          # single model + framework
./run.sh --json                              # machine-readable output
./run.sh --strict                            # controlled f32 comparison
./run.sh --warmup-runs 5 --measurement-runs 20
./run.sh --results-dir ../paper-results       # preserve a named experiment
./run.sh -f meganeura -m Whisper-tiny --profile --profile-samples 5
```

`--profile` retains the ordinary grouped-pass timing as the reported benchmark,
then asks Meganeura to collect repeated per-dispatch hardware timestamps for
inference, latency, and training. Structured JSON sidecars are written under
`<results-dir>/profiles/` and referenced by the Meganeura result artifact.
They include the selected pipeline variants, phase and kernel-family
breakdowns, workgroup geometry, driver pipeline statistics when available,
and the instrumentation-overhead ratio. Capture tools such as RenderDoc are
not required. Summarize one profile or compare two revisions with:

```bash
python3 scripts/profile_report.py results/profiles/<profile>.json
python3 scripts/profile_report.py before.json after.json --top 20
```

Inferena normally builds the Meganeura revision pinned in `Cargo.toml`. When
developing both sibling repositories, opt into the local working tree
explicitly; the result records the revision with a `-dirty` suffix when
appropriate:

```bash
INFERENA_MEGANEURA_PATH=../meganeura ./run.sh -f pytorch,meganeura
```

### Download pre-trained weights

```bash
pip install huggingface-hub
./models/download.sh SmolLM2-135M
```

Or generate random-init weights locally (no network needed):

```bash
python3 models/generate_weights.py SmolLM2-135M
```

## Project structure

```
./
├── run.sh                    # Main entry point
├── .github/workflows/ci.yml  # CI: build check + smoke test with Lavapipe
├── harness/                  # Rust: orchestration, timing, output comparison
├── frameworks/
│   ├── pytorch/              # Python + bash wrapper (HF transformers)
│   ├── burn/                 # Rust (wgpu backend, LLaMA-style model)
│   ├── luminal/              # Rust (graph-compiled, e-graph optimized)
│   └── meganeura/            # Rust (blade-graphics, graph compiled)
├── models/
│   ├── SmolLM2-135M.md       # Model description + results
│   ├── SmolVLA.md            # Model description + results
│   ├── download.sh           # HuggingFace model downloader
│   └── generate_weights.py   # Generate random-init weights locally
└── results/                  # Benchmark output (gitignored, per-run)
```

## License

MIT
