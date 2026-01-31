#!/bin/bash
# Download CLEAN weights
# - ESM-1b: https://huggingface.co/facebook/esm1b_t33_650M_UR50S
# - CLEAN: https://huggingface.co/Passion4ever/protforge-clean

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/../weights/clean"
mkdir -p "${WEIGHTS_DIR}"

# ============================================
# ESM-1b (HuggingFace)
# ============================================
ESM_DIR="${WEIGHTS_DIR}/esm1b"
mkdir -p "${ESM_DIR}"
cd "${ESM_DIR}"

HF_ESM="https://huggingface.co/facebook/esm1b_t33_650M_UR50S/resolve/32996d886e0acd61ef294ed49993cf75ce47fb7f"

echo "Downloading ESM-1b to: $(pwd)"
echo ""

download() {
    if [ -f "$1" ]; then
        printf "  %-30s (exists)\n" "$1"
    else
        printf "  %-30s" "$1"
        wget -c -q "${HF_ESM}/$1" && echo "✓"
    fi
}

download_large() {
    if [ -f "$1" ]; then
        echo "  $1 (exists)"
    else
        echo "  $1"
        wget -c -q --show-progress "${HF_ESM}/$1"
    fi
}

download "config.json"
download "vocab.txt"
download "tokenizer_config.json"
download "special_tokens_map.json"
download_large "model.safetensors"

# ============================================
# CLEAN model weights (HuggingFace)
# ============================================
cd "${WEIGHTS_DIR}"

HF_CLEAN="https://huggingface.co/Passion4ever/protforge-clean/resolve/main"

echo ""
echo "Downloading CLEAN weights to: ${WEIGHTS_DIR}/pretrained"
echo ""

if [ ! -f "pretrained/split100.pth" ]; then
    echo "  Downloading clean_pretrained.zip..."
    wget -c -q --show-progress "${HF_CLEAN}/clean_pretrained.zip"
    echo "  Extracting..."
    unzip -q -o clean_pretrained.zip
    rm -f clean_pretrained.zip
    echo "  ✓"
else
    echo "  CLEAN weights already exist, skipping."
fi

echo ""
echo "Done!"

