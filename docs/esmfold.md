# ESMFold

ESMFold structure prediction.

## Download Weights

```bash
bash scripts/download_esmfold_weights.sh
```

## Examples

```
examples/esmfold/
├── data/            # Example FASTA files
│   ├── single.fasta # Single sequence
│   ├── multi.fasta  # Multiple sequences
│   └── long.fasta   # Long sequence (for chunking)
└── outputs/         # Output directory
```

## Usage

```bash
cd examples

# Basic prediction
esmfold -i esmfold/data/single.fasta -o esmfold/outputs/single

# Multiple sequences
esmfold -i esmfold/data/multi.fasta -o esmfold/outputs/multi

# With fp16 (faster, less memory)
esmfold -i esmfold/data/single.fasta -o esmfold/outputs/single_fp16 --fp16

# Long sequence with chunking
esmfold -i esmfold/data/long.fasta -o esmfold/outputs/long --fp16 --chunk-size 64
```

## Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input FASTA file |
| `-o, --output` | Output directory |
| `--device` | Device (cuda/cuda:0/cpu) |
| `--fp16` | Use float16 precision (faster) |
| `--bf16` | Use bfloat16 precision (Ampere+ GPU) |
| `--chunk-size` | Chunk size for long sequences |
| `--num-recycles` | Number of recycles (default: 4) |
| `--weights` | Weights directory |
| `-h, --help` | Show help message |

## Output

For each sequence, ESMFold outputs:
- `{name}.pdb` - Predicted structure
