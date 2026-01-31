# ProtForge

Protein design toolkit.

## Tools

| Tool | Description | Implementation | Docs |
|------|-------------|----------------|------|
| `mpnn` | ProteinMPNN/LigandMPNN sequence design | Refactored from [LigandMPNN](https://github.com/dauparas/LigandMPNN) | [EN](docs/mpnn.md) / [中文](docs/mpnn_zh.md) |
| `esmfold` | ESMFold structure prediction | Wrapper for [HuggingFace ESMFold](https://huggingface.co/facebook/esmfold_v1) | [Docs](docs/esmfold.md) |

## Installation

```bash
# Create conda environment
conda create -n protforge python=3.10
conda activate protforge

# Install dependencies
pip install -r requirements.txt
```

Download weights for specific tools - see docs for each tool.
