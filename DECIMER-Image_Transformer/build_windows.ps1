$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$builderPackages = "C:\Users\11588\miniconda3\Lib\site-packages"
$runtimePackages = Join-Path $project ".venv\Lib\site-packages"
$python = "C:\Users\11588\miniconda3\python.exe"

Set-Location $project
$env:PYTHONPATH = "$runtimePackages;$builderPackages"
& $python -m PyInstaller --noconfirm --clean "decimer_desktop.spec"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

Write-Host "Build completed: $project\dist\DECIMER Desktop"
