#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================"
echo "    ScrappyAI Setup & Installation"
echo "========================================"

echo "[1/4] Creating and activating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/4] Installing browser binaries for Playwright..."
# Playwright is used for dynamic rendering of JS-heavy sites
playwright install chromium

echo "[4/4] Setting up models directory & downloading LLM..."
mkdir -p models

MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_PATH="models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"

if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at $MODEL_PATH. Skipping download."
else
    echo "Downloading Qwen2.5-3B-Instruct-Q4_K_M.gguf (this might take a while)..."
    wget -c -O "$MODEL_PATH" "$MODEL_URL"
fi

echo "========================================"
echo "Setup Complete!"
echo "To get started, activate the virtual environment with:"
echo "    source .venv/bin/activate"
echo "========================================"
