param(
    [switch]$InstallCudaLatest,
    [switch]$InstallCuda,
    [string]$CudaVersion = "13.2"
)

$ErrorActionPreference = "Stop"

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-Command($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-VSBuildToolsPath {
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        return $null
    }

    $path = & $vswhere `
        -products Microsoft.VisualStudio.Product.BuildTools `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath `
        -latest

    if ([string]::IsNullOrWhiteSpace($path)) {
        return $null
    }

    return $path.Trim()
}

Write-Step "Checking Windows package manager"
if (-not (Test-Command winget)) {
    throw "winget was not found. Install App Installer from Microsoft Store first."
}

Write-Step "Checking NVIDIA driver"
if (Test-Command nvidia-smi) {
    nvidia-smi
} else {
    Write-Warning "nvidia-smi was not found. Install or repair the NVIDIA driver first."
}

Write-Step "Checking Visual Studio 2022 Build Tools C++ workload"
$vsPath = Get-VSBuildToolsPath
if ($vsPath) {
    Write-Host "Found VS Build Tools with C++ tools:"
    Write-Host "  $vsPath"
} else {
    Write-Step "Installing Visual Studio 2022 Build Tools C++ workload"
    winget install `
        --id Microsoft.VisualStudio.2022.BuildTools `
        --exact `
        --accept-package-agreements `
        --accept-source-agreements `
        --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

    $vsPath = Get-VSBuildToolsPath
    if (-not $vsPath) {
        throw "VS Build Tools C++ workload was not detected after installation."
    }
}

$devShell = Join-Path $vsPath "Common7\Tools\Launch-VsDevShell.ps1"
Write-Host "Developer shell loader:"
Write-Host "  $devShell"

Write-Step "Checking CUDA Toolkit package information"
winget show --id Nvidia.CUDA --exact
winget show --id Nvidia.CUDA --exact --versions

if ($InstallCudaLatest) {
    Write-Step "Installing latest CUDA Toolkit from winget"
    winget install `
        --id Nvidia.CUDA `
        --exact `
        --accept-package-agreements `
        --accept-source-agreements
} elseif ($InstallCuda) {
    Write-Step "Installing CUDA Toolkit $CudaVersion from winget"
    winget install `
        --id Nvidia.CUDA `
        --exact `
        --version $CudaVersion `
        --accept-package-agreements `
        --accept-source-agreements
} else {
    Write-Host ""
    Write-Host "CUDA Toolkit was not installed automatically." -ForegroundColor Yellow
    Write-Host "Your current driver reports CUDA 13.2 support. Prefer installing CUDA Toolkit 13.2, or update the driver before installing a newer Toolkit."
    Write-Host "To install CUDA Toolkit 13.2, rerun:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\setup_cuda_env.ps1 -InstallCuda -CudaVersion 13.2"
    Write-Host "To let this script install winget's latest CUDA package, rerun:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\setup_cuda_env.ps1 -InstallCudaLatest"
}

Write-Step "Next checks after installation"
Write-Host "Open 'Developer PowerShell for VS 2022', cd to this folder, then run:"
Write-Host "  cl"
Write-Host "  nvcc --version"
Write-Host "  nvcc -O3 -std=c++17 -arch=sm_89 main_cuda.cu -o main_cuda.exe"
Write-Host "  .\main_cuda.exe"
