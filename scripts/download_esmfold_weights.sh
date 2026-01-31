#!/bin/bash
# Download ESMFold weights from Hugging Face
# Source: https://huggingface.co/facebook/esmfold_v1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/../weights/esmfold"
cd "${SCRIPT_DIR}/../weights/esmfold"

# Use a specific version (safetensors format)
HF_REPO="https://huggingface.co/facebook/esmfold_v1/resolve/ba837a39b67e59941c3f017d6c2a064f567038d9"

echo "Downloading to: $(pwd)"
echo ""

download() {
    printf "  %-30s" "$1"
    wget -c -q "${HF_REPO}/$1" && echo "✓"
}

download_large() {
    echo "  $1"
    wget -c -q --show-progress "${HF_REPO}/$1"
}

download "config.json"
download "vocab.txt"
download "tokenizer_config.json"
download "special_tokens_map.json"
download_large "model.safetensors"

echo ""
echo "Done!"
