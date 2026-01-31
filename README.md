# ProtForge

Protein design toolkit.

## Tools

| Tool | Description | Implementation | Docs |
|------|-------------|----------------|------|
| `mpnn` | ProteinMPNN/LigandMPNN sequence design | Refactored from [LigandMPNN](https://github.com/dauparas/LigandMPNN) | [EN](docs/mpnn.md) / [中文](docs/mpnn_zh.md) |
| `esmfold` | ESMFold structure prediction | Wrapper for [HuggingFace ESMFold](https://huggingface.co/facebook/esmfold_v1) | [Docs](docs/esmfold.md) |
| `clean` | EC number prediction | Refactored from [CLEAN](https://github.com/tttianhao/CLEAN) | [Docs](docs/clean.md) |

## Installation

```bash
# Create conda environment
conda create -n protforge python=3.10
conda activate protforge

# Install PyTorch (CUDA 12.1)
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install protforge
pip install -e ".[all]"        # All tools
# Or install specific modules
pip install -e ".[mpnn]"       # MPNN only
pip install -e ".[esmfold]"    # ESMFold only
pip install -e ".[clean]"      # CLEAN only
```

Download weights for specific tools - see docs for each tool.
