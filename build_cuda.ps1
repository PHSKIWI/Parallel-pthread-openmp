param(
    [string]$Source = "main_cuda.cu",
    [string]$Output = "main_cuda.exe",
    [string]$Arch = "sm_89"
)

$ErrorActionPreference = "Stop"

function Test-Command($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe was not found. Install Visual Studio 2022 Build Tools first."
}

$vsPath = & $vswhere `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath `
    -latest

if ([string]::IsNullOrWhiteSpace($vsPath)) {
    throw "MSVC C++ tools were not found. Install the C++ desktop/build tools workload."
}

$devShell = Join-Path $vsPath.Trim() "Common7\Tools\Launch-VsDevShell.ps1"
if (-not (Test-Path $devShell)) {
    throw "Launch-VsDevShell.ps1 was not found at $devShell"
}

& $devShell -Arch amd64 -HostArch amd64

$asciiTemp = $null
$tempCandidates = @(
    "C:\cuda_tmp",
    "C:\Users\Public\cuda_tmp",
    "C:\Windows\Temp"
)

foreach ($candidate in $tempCandidates) {
    try {
        if (-not (Test-Path $candidate)) {
            New-Item -ItemType Directory -Path $candidate -Force | Out-Null
        }

        $probe = Join-Path $candidate "nvcc_temp_probe.txt"
        Set-Content -Path $probe -Value "ok" -Encoding ASCII
        Remove-Item -Path $probe -Force
        $asciiTemp = $candidate
        break
    } catch {
        $asciiTemp = $null
    }
}

if (-not $asciiTemp) {
    throw "Could not find a writable ASCII temp directory. Run PowerShell as Administrator or create C:\cuda_tmp."
}

$env:TEMP = $asciiTemp
$env:TMP = $asciiTemp

if (-not (Test-Command nvcc)) {
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    $latestCudaBin = Get-ChildItem $cudaRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "bin" } |
        Where-Object { Test-Path (Join-Path $_ "nvcc.exe") } |
        Select-Object -First 1

    if ($latestCudaBin) {
        $env:Path = "$latestCudaBin;$env:Path"
    }
}

if (-not (Test-Command nvcc)) {
    throw "nvcc was not found. Install CUDA Toolkit and open a new terminal."
}

nvcc -O3 -std=c++17 "-arch=$Arch" -Xcompiler "/utf-8" $Source -o $Output
