# Create an isolated comparison environment, downloading Python when needed.
# Windows PowerShell variant of setup.sh.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Backend,

    [Parameter(Position = 1)]
    [string]$EnvPath
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
    { $_ -in 'cu130', 'xpu', 'cpu' } { break }
    default {
        Write-Error "Unknown wheel backend: $Backend`nUsage: powershell scripts/setup.ps1 <cu130|xpu|cpu> [new-venv-path]"
        exit 2
    }
}

if (-not $EnvPath) {
    $EnvPath = Join-Path $Root '.venv'
}
if (Test-Path $EnvPath) {
    Write-Error "Environment already exists: $EnvPath. Choose a new path; nothing was replaced."
    exit 2
}

$PythonVersion = (Get-Content (Join-Path $Root '.python-version') -TotalCount 1).Trim()
uv venv --managed-python --python $PythonVersion $EnvPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$EnvPython = Join-Path $EnvPath 'Scripts\python.exe'

$TorchArgs = @('--torch-backend', $Backend)

$Requirements = Join-Path $Root 'requirements-pytorch.txt'
if ($Backend -eq 'cu130') {
    $Requirements = Join-Path $Root 'requirements-pytorch-cu130-windows.txt'
}

uv pip install --python $EnvPython @TorchArgs -r $Requirements
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ProbeBackend = switch -Regex ($Backend) {
    '^cu130$'  { 'cuda' }
    '^rocm'    { 'rocm' }
    default    { $Backend }
}

& $EnvPython (Join-Path $Root 'scripts\check_environment.py') --backend $ProbeBackend
exit $LASTEXITCODE
