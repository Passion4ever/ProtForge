# CLEAN

CLEAN (Contrastive Learning enabled Enzyme ANnotation) predicts EC numbers for enzyme sequences.

## Download Weights

```bash
bash scripts/download_clean_weights.sh
```

This downloads both ESM-1b (~2.5GB) and CLEAN model weights.

## Examples

```
examples/clean/
└── data/
    ├── enzymes.fasta       # 5 enzyme sequences
    └── enzymes_labels.txt  # True EC labels for evaluation
```

## Usage

```bash
cd examples

# Basic prediction
clean -i clean/data/enzymes.fasta -o clean/results.csv

# With evaluation
clean -i clean/data/enzymes.fasta -o clean/results.csv --labels clean/data/enzymes_labels.txt

# Use split70 model (better for novel sequences)
clean -i clean/data/enzymes.fasta -o clean/results.csv --model 70

# Disable GMM confidence (output raw distances)
clean -i clean/data/enzymes.fasta -o clean/results.csv --no-gmm
```

## Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input FASTA file |
| `-o, --output` | Output CSV file |
| `--device` | Device (cuda/cuda:0/cpu) |
| `--model` | Model variant: 100 or 70 (default: 100) |
| `--labels` | True labels file for evaluation |
| `--no-gmm` | Disable GMM confidence |
| `--clean-weights` | CLEAN weights directory |
| `--esm-weights` | ESM-1b weights directory |

## Model Variants

- **split100** (default): 100% sequence identity clustering
- **split70**: 70% sequence identity clustering, better for novel sequences

## Output

Results are saved as CSV:
```
sequence_id,EC:number/confidence,EC:number/confidence,...
```

Example:
```
E0VIU9,EC:2.3.2.31/0.9523
Q838J7,EC:4.2.1.113/0.8891,EC:4.2.1.52/0.7234
```

## Labels File Format

For evaluation, provide a labels file with one EC per line (matching FASTA order):
```
2.3.2.31
4.2.1.113
2.7.7.18;3.6.1.41
```

Multiple ECs separated by semicolon.
