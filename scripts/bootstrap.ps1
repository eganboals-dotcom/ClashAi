param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
if (-not (Test-Path 'vendor/FirstLight_CR/native_runner')) {
    git submodule update --init --recursive
    if ($LASTEXITCODE) { throw 'Upstream checkout failed' }
}
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE) { throw 'Python 3.12 is required; pass -Python with its executable path' }
}
$venvPython = Join-Path $repo '.venv/Scripts/python.exe'
& $venvPython -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE) { throw 'CUDA PyTorch installation failed' }
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE) { throw 'Dependency installation failed' }
& $venvPython scripts/lab.py verify-model
if ($LASTEXITCODE) { throw 'Model verification failed' }
& $venvPython scripts/lab.py coverage
& $venvPython scripts/lab.py doctor
# Doctor deliberately returns nonzero while the external engine/training host is absent.
