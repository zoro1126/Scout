# ========================================
#     ScrappyAI Setup & Installation
#         Windows 10/11 Edition
# ========================================
# Run this script from the project root in PowerShell (run as Administrator if needed):
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#   .\install_deps_and_models.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    ScrappyAI Setup & Installation     " -ForegroundColor Cyan
Write-Host "         Windows 10/11 Edition         " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# Step 1: Create and activate virtual environment
# ---------------------------------------------------------------------------
Write-Host "`n[1/5] Creating virtual environment..." -ForegroundColor Yellow

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "  [ERROR] 'python' not found. Please install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

$pythonVersion = python --version 2>&1
Write-Host "  [INFO] Found: $pythonVersion" -ForegroundColor Green

python -m venv .venv

Write-Host "  [INFO] Activating virtual environment..." -ForegroundColor Green
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "  [ERROR] Virtual environment activation script not found at $activateScript" -ForegroundColor Red
    exit 1
}
. $activateScript

# ---------------------------------------------------------------------------
# Step 2: Install Python dependencies
# ---------------------------------------------------------------------------
Write-Host "`n[2/5] Installing Python dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Step 3: Detect GPU / CUDA support
# ---------------------------------------------------------------------------
Write-Host "`n[3/5] Detecting GPU / CUDA support..." -ForegroundColor Yellow

$gpuDetected = $false
try {
    $nvidiaSmi = & nvidia-smi -L 2>$null
    if ($nvidiaSmi -match "GPU") {
        $gpuDetected = $true
    }
} catch {
    $gpuDetected = $false
}

if ($gpuDetected) {
    Write-Host "  [GPU] NVIDIA GPU detected — rebuilding llama-cpp-python with CUDA support..." -ForegroundColor Green
    $env:CMAKE_ARGS = "-DGGML_CUDA=ON"
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    Write-Host "  [GPU] llama-cpp-python rebuilt with CUDA. Inference will run on GPU automatically." -ForegroundColor Green
} else {
    Write-Host "  [CPU] No NVIDIA GPU detected — using CPU-only llama-cpp-python (already installed)." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# Step 4: Install Playwright browser binaries
# ---------------------------------------------------------------------------
Write-Host "`n[4/5] Installing browser binaries for Playwright..." -ForegroundColor Yellow
playwright install chromium

# ---------------------------------------------------------------------------
# Step 5: Download the quantized LLM
# ---------------------------------------------------------------------------
Write-Host "`n[5/5] Setting up models directory & downloading LLM..." -ForegroundColor Yellow

$modelsDir = "backend\models"
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
    Write-Host "  [INFO] Created directory: $modelsDir" -ForegroundColor Green
}

$modelUrl  = "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
$modelPath = "$modelsDir\Qwen2.5-3B-Instruct-Q4_K_M.gguf"

if (Test-Path $modelPath) {
    Write-Host "  [INFO] Model already exists at $modelPath. Skipping download." -ForegroundColor Green
} else {
    Write-Host "  [INFO] Downloading Qwen2.5-3B-Instruct-Q4_K_M.gguf (~2.4 GB, this may take a while)..." -ForegroundColor Yellow

    # Prefer curl.exe (ships with Windows 10 1803+) over PowerShell's Invoke-WebRequest
    # because Invoke-WebRequest buffers the entire file in memory before writing.
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        curl.exe -L -C - -o $modelPath $modelUrl
    } else {
        Write-Host "  [INFO] curl.exe not found — falling back to Invoke-WebRequest (slower for large files)..." -ForegroundColor Yellow
        # Disable progress bar for significantly faster downloads via Invoke-WebRequest
        $prevProgressPreference = $ProgressPreference
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
        $ProgressPreference = $prevProgressPreference
    }

    Write-Host "  [INFO] Model downloaded successfully." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "  To get started, activate the virtual environment with:" -ForegroundColor White
Write-Host "      .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  Then run the app:" -ForegroundColor White
Write-Host "      python backend\web_ui.py" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
