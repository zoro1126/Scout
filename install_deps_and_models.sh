#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================"
echo "    ScrappyAI Setup & Installation"
echo "========================================"

echo "[1/5] Creating and activating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/5] Detecting GPU / CUDA support..."
if command -v nvidia-smi &> /dev/null && nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
    echo "  [GPU] NVIDIA GPU detected — rebuilding llama-cpp-python with CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    echo "  [GPU] llama-cpp-python rebuilt with CUDA. Inference will run on GPU automatically."
else
    echo "  [CPU] No NVIDIA GPU detected — using CPU-only llama-cpp-python (already installed)."
fi

echo "[4/5] Installing browser binaries for Playwright..."
# Playwright is used for dynamic rendering of JS-heavy sites
playwright install chromium

echo "[5/5] Setting up models directory & downloading LLM..."
mkdir -p backend/models

MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
MODEL_PATH="backend/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"

if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at $MODEL_PATH. Skipping download."
else
    echo "Downloading Qwen2.5-3B-Instruct-Q4_K_M.gguf (this might take a while)..."
    wget -c -O "$MODEL_PATH" "$MODEL_URL"
fi

echo "========================================"
echo "Setup Complete!"
echo "To get started, activate the virtual environment(if not already activated) with:"
echo "    source .venv/bin/activate"
echo "========================================"
