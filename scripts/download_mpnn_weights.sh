#!/bin/bash
# Download ProteinMPNN/LigandMPNN weights
# Source: https://github.com/dauparas/LigandMPNN

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/../weights/mpnn"
cd "${SCRIPT_DIR}/../weights/mpnn"

BASE_URL="https://files.ipd.uw.edu/pub/ligandmpnn"

echo "Downloading to: $(pwd)"
echo ""

download() {
    printf "  %-45s" "$1"
    wget -c -q "${BASE_URL}/$1" && echo "✓"
}

echo "ProteinMPNN"
download "proteinmpnn_v_48_002.pt"
download "proteinmpnn_v_48_010.pt"
download "proteinmpnn_v_48_020.pt"
download "proteinmpnn_v_48_030.pt"

echo "SolubleMPNN"
download "solublempnn_v_48_002.pt"
download "solublempnn_v_48_010.pt"
download "solublempnn_v_48_020.pt"
download "solublempnn_v_48_030.pt"

echo "LigandMPNN"
download "ligandmpnn_v_32_005_25.pt"
download "ligandmpnn_v_32_010_25.pt"
download "ligandmpnn_v_32_020_25.pt"
download "ligandmpnn_v_32_030_25.pt"

echo "Sidechain Packer"
download "ligandmpnn_sc_v_32_002_16.pt"

echo "Membrane MPNN"
download "global_label_membrane_mpnn_v_48_020.pt"
download "per_residue_label_membrane_mpnn_v_48_020.pt"

echo ""
echo "Done!"
