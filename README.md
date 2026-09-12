# inferena

[![CI](https://github.com/kvark/inferena/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/inferena/actions/workflows/ci.yml)

Single-GPU ML systems benchmark for matched inference and training workloads.

This experiment branch adds validated whole-phase CUDA Graph replay to the
PyTorch comparison and pins Meganeura to merged main. See the
[protocol, controls and scaling results](EXPERIMENT.md). Generated records
stay outside Git; submitted results remain reproducible at `paper-arxiv-1`.

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
| PyTorch | CUDA, ROCm, XPU, CPU | MPS, CPU | CUDA, XPU, CPU |
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

The paired runners also support pinned **SmolLM2-360M and SmolLM2-1.7B** base
checkpoints. These opt-in scaling workloads are not part of the historical
results table. Use `--inference-only` for the same forward-only protocol across
sizes when f32 training does not fit; absent training is not a validated result.
The measured 1.7B workload exposes host-backed buffer placement on the RTX 5070;
see the experiment notes before treating it as an all-VRAM scaling point.

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
- **uv** for automatic installation of the Python version in `.python-version`
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
# sudo apt install libze-intel-gpu1 libze1 intel-opencl-icd intel-ocloc
# intel-opencl-icd supplies the shared IGC compiler used by Level Zero; this
# benchmark does not submit OpenCL work. intel-ocloc is needed by Triton XPU.
# SYCL/oneMKL runtime libs ship as dependencies of the torch+xpu wheel.

# Enable torch.compile on CPU (Inductor needs Python headers + g++):
# sudo apt install python3-dev g++
# Requested compilation fails explicitly if these are missing; no eager fallback.
```

Run `./run.sh --check` after setup to verify what's working and get
install hints for anything missing.

## Quick start

The collection source is maintained on one branch while the protocol settles.
The provisional collection tags were removed; use the same full branch revision
on every machine, and check the current readiness finding before collecting:

```sh
git fetch origin
git switch experiment/p3hpc-cuda-graphs
git pull --ff-only
```

The [readiness findings](EXPERIMENT.md#current-collection-readiness-september-12)
explain the precision-aware replay checks and preserved failed attempts.
Cross-engine accuracy gates are unchanged.

For the paired P3HPC campaign, install [uv](https://docs.astral.sh/uv/getting-started/installation/)
once, then let the setup script install Python and the pinned requirements:

```bash
bash scripts/setup.sh cu130                 # new .venv-p3hpc; Python downloaded automatically
# Other wheel backends: xpu, rocm7.2, cpu; mps uses the macOS PyPI wheel.
.venv-p3hpc/bin/python scripts/p3hpc.py
```

On native Windows, from PowerShell:

```powershell
.\scripts\setup.ps1 cu130
.\.venv-p3hpc\Scripts\python.exe scripts\p3hpc.py
```

After activating that environment, the collection command on every platform is
simply **`python scripts/p3hpc.py`**. Run it from a clean checkout of the same
collection revision on each machine. It prepares missing pinned 135M weights,
or verifies/adopts an exact legacy cache, detects the reference backend and
matching native GPU, and validates all five common models in both precision
classes inside three fresh-process measurement replicates
per condition (5 warmups, 20 samples). No CPU/eager fallback, automatic model
exclusion, discarded sample, or retry is allowed. Strict gradients must pass
the 5% gate in every process. Accelerated gradients retain every sample under
a 10% safety ceiling, then require the median cross-engine error to remain
below 5% across the three processes. Every raw result remains in the campaign.
The full campaign can take a while because compilation caches are private to
each of its 90 reference processes.

When stock PyTorch max-autotune fails on an AMD consumer GPU, preserve that
failure as a portability result, then collect the runnable conditions as an
explicitly labelled availability dataset:

```bash
bash scripts/setup.sh rocm7.2 --no-max-autotune
.venv-p3hpc/bin/python scripts/p3hpc.py --no-max-autotune
```

This is not the full reference-condition matrix. The manifest records the
declared, selected and omitted conditions and marks its coverage as
`availability-subset`. See the [Radeon 780M case study](EXPERIMENT.md#radeon-780m-availability-case).

The printed `../inferena-results/<host>-<UTC>-<source>/` directory contains the
manifest, records and logs. Keep that whole directory; `campaign.json` must say
`"status": "complete"` before treating it as a complete cohort. Nothing is
overwritten or added to Git. Optional `--qualify-only --models ResNet-50
--precisions strict` provides a short new-device check; it does not collect
publication timings. Larger SmolLM2 sizes are a separate opt-in
[scaling campaign](EXPERIMENT.md#matched-smollm2-scaling).

Windows still needs Git for Windows: the Python collector locates Git Bash for
the underlying runners. CUDA setup also installs the matching Windows Triton
compiler; it is not included by the PyTorch wheel. Existing environments are
never replaced; pass a second argument to choose a new path. Setup verifies the
common PyTorch source and runs a small
forward/backward probe in each reference mode, including CUDA Graph replay.
GPU drivers and Rust's platform build tools remain system prerequisites.
See [Windows collection](EXPERIMENT.md#windows-nvidia) and
[XPU and Nsight instructions](EXPERIMENT.md).

Requirements files cannot select/install an interpreter by themselves. For
the broader, non-paper runner dependencies, `uv venv .venv` also reads
`.python-version` and downloads Python, then install the desired requirements:

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements-nvidia.txt    # NVIDIA CUDA; other vendor files are separate cohorts
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
