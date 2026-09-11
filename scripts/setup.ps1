# Create an isolated comparison environment, downloading Python when needed.
# Windows PowerShell variant of setup.sh.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Backend,

    [Parameter(Position = 1)]
    [string]$EnvPath,

    [switch]$NoMaxAutotune
)

$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 2
}

switch ($Backend) {
    { $_ -in 'cu130', 'xpu', 'cpu', 'mps', 'rocm7.2' } { break }
    default {
        Write-Error "Unknown wheel backend: $Backend`nUsage: powershell scripts/setup.ps1 <cu130|xpu|cpu|mps|rocm7.2> [new-venv-path]"
        exit 2
    }
}

if (-not $EnvPath) {
    $EnvPath = Join-Path $Root '.venv-p3hpc'
}
if (Test-Path $EnvPath) {
    Write-Error "Environment already exists: $EnvPath. Choose a new path; nothing was replaced."
    exit 2
}

$PythonVersion = (Get-Content (Join-Path $Root '.python-version') -TotalCount 1).Trim()
uv venv --managed-python --python $PythonVersion $EnvPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$EnvPython = Join-Path $EnvPath 'Scripts\python.exe'

$TorchArgs = @()
if ($Backend -ne 'mps') {
    $TorchArgs = @('--torch-backend', $Backend)
}

$Requirements = Join-Path $Root 'requirements-p3hpc.txt'
if ($Backend -eq 'cu130') {
    $Requirements = Join-Path $Root 'requirements-p3hpc-cu130-windows.txt'
}

uv pip install --python $EnvPython @TorchArgs -r $Requirements
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ProbeBackend = switch -Regex ($Backend) {
    '^cu130$'  { 'cuda' }
    '^rocm'    { 'rocm' }
    default    { $Backend }
}

$ProbeArgs = @('--backend', $ProbeBackend)
if ($NoMaxAutotune) {
    $ProbeArgs += '--no-max-autotune'
}
& $EnvPython (Join-Path $Root 'scripts\check_environment.py') @ProbeArgs
exit $LASTEXITCODE
