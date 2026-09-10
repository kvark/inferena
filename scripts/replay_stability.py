"""CUDA diffusion diagnostic: ordinary repeat variability versus capture gates.

Not a paired qualification or performance campaign. Use the pinned environment.
Controls: no arguments, --deterministic (cuDNN), --all-deterministic (PyTorch).
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'frameworks/pytorch'))
import bench
import torch
from p3hpc import check_torch_identity

check_torch_identity(torch.__version__, torch.version.git_version, torch.__version__)
os.environ.update(INFERENA_TORCH_BACKEND='cuda', INFERENA_TORCH_MODE='default',
                  INFERENA_CUDA_GRAPHS='1', INFERENA_STRICT='1',
                  INFERENA_WARMUP_RUNS='5', INFERENA_MEASUREMENT_RUNS='1',
                  NVIDIA_TF32_OVERRIDE='0', INFERENA_REQUIRE_LOCAL_WEIGHTS='1')
all_deterministic = '--all-deterministic' in sys.argv
deterministic = '--deterministic' in sys.argv or all_deterministic
torch.backends.cudnn.deterministic = deterministic
if all_deterministic:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
destination = Path(tempfile.mkdtemp(prefix='inferena-replay-stability-', dir=ROOT.parent))
print(f'Diagnostic: {destination}', file=sys.stderr, flush=True)
report = {
    'kind': 'uncaptured-repeat-diagnostic-not-qualification',
    'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'torch_version': torch.__version__, 'torch_source': torch.version.git_version,
    'cudnn_deterministic': deterministic, 'all_deterministic': all_deterministic, 'repeats': [],
}
capture = bench.capture_phase


def checked_capture(fn, model=None, stream=None):
    if model is not None:
        reference = None
        for index in range(8):
            model.zero_grad(set_to_none=True)
            outputs = fn()
            torch.cuda.synchronize()
            gradients = {name: parameter.grad.detach().cpu().clone()
                         for name, parameter in model.named_parameters() if parameter.grad is not None}
            if reference is None:
                reference = gradients
            row = {'repeat': index + 1, 'failed_tensors': []}
            for name, actual in gradients.items():
                expected = reference[name]
                mismatches = int((~torch.isclose(actual, expected, rtol=1e-4, atol=1e-6)).sum())
                if mismatches:
                    row['failed_tensors'].append({'name': name, 'mismatched': mismatches,
                                                 'max_abs': float((actual - expected).abs().max())})
            report['repeats'].append(row)
            print('UNCAPTURED', json.dumps(row), file=sys.stderr, flush=True)
            del outputs
    return capture(fn, model, stream)


bench.capture_phase = checked_capture
try:
    bench.bench('StableDiffusion', bench.MODEL_REGISTRY['StableDiffusion'])
    report['capture'] = 'passed-after-extra-warmups-not-a-retry-result'
except BaseException as error:
    report['capture'] = repr(error)
    raise
finally:
    (destination / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
