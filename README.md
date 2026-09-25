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
   (seconds). The JSON describes the scope: Meganeura includes parameter/input
   upload and optional kernel tuning; PyTorch includes first specializations
   of the requested phases. PyTorch reports capture and validation separately.
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
  Meganeura f16 cooperative-matrix paths are disabled. Native f32 cooperative
  matrices remain allowed.

The two configurations are separate comparison classes because TF32 and f16
are not numerically equivalent.

PyTorch uses `torch.compile` in `default` mode, including on Windows and MPS.
CUDA, HIP and XPU runs use explicit whole-phase graph replay by default.
MPS and CPU have no equivalent replay API here. Before timing, the runner
checks repeated outputs and every participating gradient, then checks replay
against ordinary execution. This is a repeatability check, not an eager oracle.
Cross-engine validation still uses output samples and canonical gradient norms.

A requested compiler or backend failure is an error, never an eager/CPU timing
under the same label. A training qualification failure preserves already-qualified
inference as `status: partial`; failed and unattempted phases have null timings.
Runtime failures such as OOM and device loss remain errors with their tracebacks.
For diagnosis, request eager explicitly in a separate results directory:

```bash
INFERENA_TORCH_MODE=eager INFERENA_GRAPH_REPLAY=0 INFERENA_REFERENCE_DIAGNOSTIC=1 \
  ./run.sh -m Whisper-tiny -f pytorch --results-dir ../eager-diagnostic
```

`INFERENA_TORCH_MODE=max-autotune` opts into PyTorch's longer search.
`MEGANEURA_TUNE=1` opts into Meganeura's bounded kernel tuner.
Both choices are recorded. Main does not impose the paper's tuning deadlines,
multi-plan measured search, campaign replication or vendor-specific workarounds.
`INFERENA_SDPA=math|efficient|auto` is an explicit diagnostic choice, not a
hidden per-device default.

## Prerequisites

- **Rust** toolchain (for Candle, Burn, Luminal, Meganeura)
- **Python** (managed automatically by the setup scripts below)
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
# Missing compiler dependencies are reported as errors, not eager timings.
```

Run `./run.sh --check` after setup to verify what's working and get
install hints for anything missing.

## Quick start

For a PyTorch/Meganeura comparison, install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```bash
bash scripts/setup.sh cu130                 # or rocm7.2, xpu, mps, cpu
source .venv/bin/activate
python scripts/prepare_models.py SmolLM2-135M
./run.sh -f pytorch,meganeura
```

Setup downloads the Python version in `.python-version`, installs only the
comparison dependencies, and probes compiled forward/backward on the requested
backend. It refuses to overwrite an existing environment; pass a new path as
its second argument to keep multiple vendor environments. Setup does not
install GPU drivers or CUDA/HIP SDKs.

On Windows, use PowerShell for setup and Git Bash for the harness:

```powershell
.\scripts\setup.ps1 cu130
.\.venv\Scripts\python.exe scripts\prepare_models.py SmolLM2-135M
```

Then run `bash run.sh -f pytorch,meganeura`. The wrapper finds `.venv`
automatically. An explicit `PYTHON` takes precedence over an activated venv.

For other frameworks, the broader existing requirements files are still
available:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-nvidia.txt       # NVIDIA CUDA
# pip install -r requirements-amd.txt        # AMD ROCm
# pip install -r requirements-intel.txt      # Intel XPU (Arc / Xe iGPU)
# pip install -r requirements-apple.txt      # Apple Metal
# pip install -r requirements-cpu.txt        # CPU only
python scripts/prepare_models.py SmolLM2-135M
./run.sh                                     # practical defaults, all models/frameworks
./run.sh -m SmolLM2-135M                     # single model
./run.sh -m SmolLM2-135M -f pytorch          # single model + framework
./run.sh --json                              # machine-readable output
./run.sh --strict                            # controlled f32 comparison
./run.sh --warmup-runs 5 --measurement-runs 20
./run.sh --results-dir ../paper-results       # preserve a named experiment
./run.sh -f meganeura -m Whisper-tiny --profile --profile-samples 5
```

`--profile` retains the ordinary timing as the reported benchmark,
then asks Meganeura to collect repeated GPU pass timestamps for
inference, latency, and training. Structured JSON sidecars are written under
`<results-dir>/profiles/` and referenced by the Meganeura result artifact.
They include the selected pipeline variants, phase and kernel-family
breakdowns, workgroup geometry, driver pipeline statistics when available,
and the instrumentation-overhead ratio. Capture tools such as RenderDoc are
not required. PyTorch writes a separate profiler timeline after its ordinary
measurements. GPU pass intervals are not kernel-only time, and subtracting
their sum from wall time does not measure CPU overhead or barriers.
Summarize one Meganeura profile or compare two revisions with:

```bash
python3 scripts/profile_report.py results/profiles/<profile>.json
python3 scripts/profile_report.py before.json after.json --top 20
```

For Nsight Systems, choose a fresh directory and one model per invocation:

```bash
mkdir -p ../nsys-smollm
INFERENA_NSYS="$(command -v nsys)" INFERENA_NSYS_DIR="$(cd ../nsys-smollm && pwd)" \
  ./run.sh -m SmolLM2-135M -f pytorch,meganeura --results-dir ../nsys-smollm
```

Those runs are marked diagnostic, excluded from charts, and cannot use
`--update`. Their timings must not replace ordinary measurements.

### Select a backend or larger model

Set `INFERENA_TORCH_BACKEND=cuda|rocm|xpu|mps|cpu` to require that backend;
it cannot fall back to CPU. Without it, the runner keeps automatic discovery.
On multi-GPU machines, use the vendor's visibility variables for PyTorch and
`MEGANEURA_DEVICE_ID` for Meganeura. List native adapters with:

```bash
cargo run --release --locked -p inferena-meganeura -- --list-devices
```

The wrapper no longer hides non-NVIDIA Vulkan drivers. Use
`--allow-integrated-gpu` to disable the harness's discrete-GPU preference.

The larger SmolLM2 models are opt-in, not part of `--model all`:

```bash
python scripts/prepare_models.py SmolLM2-360M SmolLM2-1.7B
./run.sh -m SmolLM2-360M -f pytorch,meganeura
./run.sh -m SmolLM2-1.7B -f pytorch,meganeura --inference-only
```

`--inference-only` avoids the training allocation and reports training as
unrequested, not as zero time. `INFERENA_WARMUP_SECONDS` optionally adds a
minimum warmup duration to both runners; the default remains five warmup calls.

Inferena normally builds the Meganeura revision pinned in `Cargo.toml`. When
developing both sibling repositories, opt into the local working tree
explicitly; the result records the revision with a `-dirty` suffix when
appropriate:

```bash
INFERENA_MEGANEURA_PATH=../meganeura ./run.sh -f pytorch,meganeura
```

### Download pre-trained weights

```bash
python scripts/prepare_models.py SmolLM2-135M
```

This downloads an immutable revision, verifies file hashes and writes
`models/<name>/source.json`. Existing mismatched files are rejected, not
overwritten. `./run.sh --download` uses the same helper. SmolLM2-360M now uses
the base checkpoint, not the old downloader's Instruct checkpoint; move old
weights aside explicitly before preparing that model.

Or generate random-init weights locally (no network needed):

```bash
python3 models/generate_weights.py SmolLM2-135M
```

Synthetic SmolVLA, diffusion, ResNet and Whisper parameters keep main's
`name-seeded-sine-v1` initializer across adapters. They do not use the paper
branch's different uniform fixture. Reproduce the paper with the
`paper-p3hpc-2026-final` tag, not main.

### Fast regression checks

```bash
cargo test --locked -p inferena-harness
python -m unittest discover -s frameworks/pytorch -p 'test_*.py'
python scripts/check_environment.py --backend cpu
```

These use small CPU fixtures, not model benchmarks. CI also publishes Python
coverage for the runners. GPU replay remains a backend-specific qualification;
run the environment probe with your actual backend before collecting data.

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
