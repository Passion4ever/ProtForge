# MPNN

ProteinMPNN/LigandMPNN sequence design.

## Download Weights

```bash
bash scripts/download_mpnn_weights.sh
```

## Usage

```bash
# Using config file
mpnn design -c config.yaml

# Using CLI arguments
mpnn design --pdb_path input.pdb --out_folder output/
```

## Examples

```
examples/mpnn/
├── configs/         # YAML configs
├── data/            # Example PDB files
└── outputs/         # Output directory
```

| Config | Description |
|--------|-------------|
| `01_basic_monomer.yaml` | Basic monomer design |
| `02_design_specific_chains.yaml` | Design specific chains in complex |
| `03_fix_residues.yaml` | Fix specific residues |
| `04_redesign_specific_residues.yaml` | Redesign only specific residues |
| `05_homooligomer.yaml` | Homo-oligomer design |
| `06_bias_amino_acids.yaml` | Bias amino acid composition |
| `07_omit_amino_acids.yaml` | Omit specific amino acids |
| `08_with_dna_ligand.yaml` | Design with DNA ligand (LigandMPNN) |
| `09_diverse_sequences.yaml` | Generate diverse sequences |
| `10_with_sidechain_packing.yaml` | Design + sidechain packing |
| `11_per_residue_bias.yaml` | Per-residue amino acid bias |
| `12_soluble_model.yaml` | Soluble protein model |

---

## Parameters

### Overview

| Category | Count | Description |
|----------|-------|-------------|
| Required | 2 | Input/Output |
| Common | 15 | Daily usage |
| Advanced | 20+ | Specific scenarios |

---

### 1. Required Parameters

#### pdb_path / pdb_dir

Input PDB file(s).

```bash
# Single file
--pdb_path ./protein.pdb

# Batch directory
--pdb_dir ./backbones/
--recursive              # Optional: recursive subdirectories
```

```yaml
# Single file
pdb_path: ./protein.pdb

# Batch directory
pdb_dir: ./backbones/
recursive: true
```

#### out_folder

Output directory.

```bash
--out_folder ./outputs/
```

Output structure:
```
outputs/
├── seqs/           # Designed sequences (.fa)
├── backbones/      # Backbone files (optional)
└── packed/         # Sidechain packing results (optional)
```

---

### 2. Model Selection

#### model_type

```bash
--model_type ligand_mpnn
```

| Value | Description | Use Case |
|-------|-------------|----------|
| `protein_mpnn` | Standard ProteinMPNN (default) | General protein design |
| `ligand_mpnn` | Ligand-aware context | DNA/RNA/small molecule/metal binding proteins |
| `soluble_mpnn` | Optimized for solubility | Proteins requiring high solubility |
| `per_residue_label_membrane_mpnn` | Membrane (per-residue labels) | Membrane proteins with buried/interface residue labels |
| `global_label_membrane_mpnn` | Membrane (global label) | Membrane proteins with only transmembrane flag |

**Decision tree:**
```
What type of protein?
│
├─ Binds ligand (DNA/RNA/small molecule/metal)
│   └─ ligand_mpnn
│
├─ Membrane protein
│   ├─ Know which residues are buried/interface → per_residue_label_membrane_mpnn
│   └─ Only know it's transmembrane → global_label_membrane_mpnn
│
├─ Need high solubility
│   └─ soluble_mpnn
│
└─ General protein
    └─ protein_mpnn (default)
```

---

### 3. Chain and Residue Control

#### chains_to_design

Specify chains to design. If not specified, all chains are designed.

```bash
--chains_to_design "A,B"
```

Format: **comma-separated, no spaces**

| Input | Meaning |
|-------|---------|
| `A` | Design chain A only |
| `A,B` | Design chains A and B |
| (not specified) | Design all chains |

#### fixed_residues

Residues to keep fixed (not designed).

```bash
--fixed_residues "A:1-10,20 B:5-15,25"
```

**Format:**
- **Space** separates different chains
- **Colon** separates chain name and residues
- **Comma** separates residues/ranges
- **Hyphen** indicates range

**Examples:**

| Input | Meaning |
|-------|---------|
| `A:1-10` | Chain A residues 1-10 |
| `A:1-10,20,30` | Chain A residues 1-10, 20, 30 |
| `A:1-10 B:5-15` | Chain A 1-10 and Chain B 5-15 |

**Note:** Ranges must be ascending: `1-10` ✓, `10-1` ✗

#### redesigned_residues

Specify **only** these residues to design, fix all others.

```bash
--redesigned_residues "A:50-60 B:20,25"
```

Format same as `fixed_residues`.

**Note:** `redesigned_residues` and `fixed_residues` are mutually exclusive. If both specified, `redesigned_residues` takes priority.

#### parse_these_chains_only

Only load specified chains, completely ignore others.

```bash
--parse_these_chains_only "A,B"
```

**Difference from chains_to_design:**

| Parameter | Effect |
|-----------|--------|
| `chains_to_design` | Which chains to **design sequences** for |
| `parse_these_chains_only` | Which chains are **loaded into model** |

**Example (protein has chains A, B, C, D):**
```bash
# Scenario 1: Design A, use BCD as context
--chains_to_design "A"
# Loads ABCD, designs A, BCD fixed but provide structural context

# Scenario 2: Only consider AB, ignore CD completely
--parse_these_chains_only "A,B"
--chains_to_design "A"
# Only loads AB, designs A, B as context, CD doesn't exist
```

---

### 4. Symmetry Control

#### homo_oligomer

Homo-oligomer mode. All chains at the same position will be designed with the same amino acid.

```bash
--homo_oligomer
```

```yaml
homo_oligomer: true
```

For: homo-dimers, trimers, etc. where all chains have identical sequences.

#### symmetry_residues

Manually specify symmetric residue groups. Residues within a group will be designed with the same amino acid.

```bash
--symmetry_residues "A12,B12|A13,B13|A14,B14"
```

**Format:**
- Pipe `|` separates different groups
- Comma `,` separates residues within a group

#### symmetry_weights

Weights for symmetric residue groups. Controls each position's influence on design decisions.

```bash
--symmetry_weights "0.5,0.5|0.5,0.5"
```

Format corresponds to `symmetry_residues`.

---

### 5. Sampling Parameters

#### temperature

Sampling temperature. Controls sequence diversity.

```bash
--temperature 0.1
```

| Value | Effect |
|-------|--------|
| 0.1 | Conservative, low diversity (default) |
| 0.2-0.5 | Medium diversity |
| 1.0+ | Aggressive, high diversity |

#### batch_size

Number of sequences per batch.

```bash
--batch_size 10
```

#### number_of_batches

Number of batches.

```bash
--number_of_batches 5
```

**Total sequences = batch_size × number_of_batches**

#### seed

Random seed for reproducibility.

```bash
--seed 42
```

---

### 6. Amino Acid Preferences

#### bias_AA

Global amino acid bias. Positive values increase probability, negative values decrease.

```bash
--bias_AA "D:1.5,E:1.5,K:1.5,R:1.5,C:-3.0"
```

Format: `AA:value`, comma-separated.

**Value reference (log-odds):**

| Value | Effect |
|-------|--------|
| +3.0 | Probability ~×20 |
| +2.0 | Probability ~×7 |
| +1.0 | Probability ~×2.7 |
| 0 | No effect |
| -1.0 | Probability ~÷2.7 |
| -3.0 | Probability ~÷20 |
| -10.0 | Effectively disabled |

**Common use cases:**
```bash
# Increase solubility (prefer charged amino acids)
--bias_AA "D:1.5,E:1.5,K:1.5,R:1.5"

# Avoid oxidation-sensitive amino acids
--bias_AA "C:-3.0,M:-2.0"

# Prefer small amino acids
--bias_AA "G:1.0,A:1.0,S:1.0"
```

#### omit_AA

Completely disable certain amino acids.

```bash
--omit_AA "CM"
```

Format: concatenated single letters, no separator.

Example: `"CM"` = disable Cysteine (C) and Methionine (M)

#### bias_AA_per_residue

Per-residue amino acid bias via JSON file.

```bash
--bias_AA_per_residue ./bias.json
```

JSON format:
```json
{
  "A12": {"P": 10.8, "G": -0.3},
  "A13": {"A": 1.5, "H": -2.0}
}
```

#### omit_AA_per_residue

Per-residue amino acid exclusion via JSON file.

```bash
--omit_AA_per_residue ./omit.json
```

JSON format:
```json
{
  "A12": "CM",
  "A13": "PG"
}
```

---

### 7. Other Common Parameters

#### verbose

Verbose output mode.

```bash
--verbose
```

```yaml
verbose: true
```

#### pack_side_chains

Enable sidechain packing.

```bash
--pack_side_chains
```

```yaml
pack_side_chains: true
```

#### file_ending

Output filename suffix.

```bash
--file_ending "_v1"
```

Result: `protein_v1.fa`

#### skip_existing

Skip PDBs with existing output (YAML batch mode only).

```yaml
skip_existing: true
```

---

### 8. LigandMPNN Parameters

When using `--model_type ligand_mpnn`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ligand_mpnn_use_atom_context` | 1 | Use ligand atom context |
| `--ligand_mpnn_cutoff_for_score` | 8.0 | Ligand context cutoff distance (Å) |
| `--ligand_mpnn_use_side_chain_context` | 0 | Use sidechain atoms as context |

---

### 9. Membrane Protein Parameters

#### per_residue_label_membrane_mpnn

```bash
--transmembrane_buried "A:20-45 B:20-45"      # Transmembrane buried residues
--transmembrane_interface "A:15-19,46-50"     # Transmembrane interface residues
```

Format same as `fixed_residues`.

**Residue label meanings:**
- **buried:** Completely inside lipid bilayer
- **interface:** At membrane-water interface
- **exposed:** In aqueous phase (default, no need to specify)

#### global_label_membrane_mpnn

```bash
--global_transmembrane_label 1   # 1=transmembrane protein, 0=soluble protein
```

---

### 10. Sidechain Packing Parameters

Available when `--pack_side_chains` is enabled:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--number_of_packs_per_design` | 4 | Packing iterations per design |
| `--sc_num_denoising_steps` | 3 | Denoising steps |
| `--sc_num_samples` | 16 | Number of samples |
| `--repack_everything` | 0 | Repack all residues |
| `--pack_with_ligand_context` | 1 | Use ligand context for packing |
| `--packed_suffix` | `_packed` | Packed PDB file suffix |
| `--force_hetatm` | flag | Force ligand atoms as HETATM |

---

### 11. Other Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--zero_indexed` | flag | Output sequence numbering from 0 (default from 1) |
| `--save_stats` | flag | Save sampling statistics to .pt file |
| `--fasta_seq_separation` | str | Multi-chain sequence separator in FASTA (default `:`) |
| `--parse_atoms_with_zero_occupancy` | flag | Parse atoms with occupancy=0 |

---

## YAML Config Examples

### Basic

```yaml
pdb_path: ./protein.pdb
out_folder: ./outputs/
model_type: protein_mpnn
temperature: 0.1
batch_size: 10
number_of_batches: 5
seed: 42
```

### Fix Partial Residues

```yaml
pdb_dir: ./backbones/
out_folder: ./outputs/
model_type: protein_mpnn
temperature: 0.1
batch_size: 10
chains_to_design: "A,B"
fixed_residues: "A:1-10,50 B:1-10"
```

### Homo-oligomer

```yaml
pdb_dir: ./backbones/
out_folder: ./outputs/
model_type: protein_mpnn
homo_oligomer: true
temperature: 0.1
batch_size: 10
```

### With Bias

```yaml
pdb_dir: ./backbones/
out_folder: ./outputs/
model_type: ligand_mpnn
temperature: 0.2
batch_size: 10
chains_to_design: "A"
bias_AA: "D:1.5,E:1.5,K:1.5,R:1.5"
omit_AA: "CM"
```

### Full Example

```yaml
pdb_dir: ./backbones/
recursive: false
out_folder: ./outputs/
skip_existing: true

model_type: ligand_mpnn
temperature: 0.1
batch_size: 10
number_of_batches: 5
seed: 42

chains_to_design: "A,B"
fixed_residues: "A:1-20 B:1-20"

bias_AA: "C:-10.0,M:-3.0"
omit_AA: "C"

pack_side_chains: true
verbose: true
```

---

## Parameter Conflicts

### Residue Control

| Conflict | Priority |
|----------|----------|
| `fixed_residues` vs `redesigned_residues` | `redesigned_residues` takes priority |
| `chains_to_design` vs `fixed_residues` | Both apply (intersection) |

### Symmetry

| Conflict | Behavior |
|----------|----------|
| `homo_oligomer` vs `symmetry_residues` | `homo_oligomer` overrides manual settings |

---

## Appendix: Amino Acid Codes

| Code | Amino Acid | Property |
|------|------------|----------|
| A | Alanine | Hydrophobic |
| C | Cysteine | Polar (disulfide bonds, oxidation-sensitive) |
| D | Aspartic acid | Acidic (negative charge) |
| E | Glutamic acid | Acidic (negative charge) |
| F | Phenylalanine | Hydrophobic/Aromatic |
| G | Glycine | Special (smallest, flexible) |
| H | Histidine | Basic/Aromatic |
| I | Isoleucine | Hydrophobic |
| K | Lysine | Basic (positive charge) |
| L | Leucine | Hydrophobic |
| M | Methionine | Hydrophobic (oxidation-sensitive) |
| N | Asparagine | Polar |
| P | Proline | Special (helix breaker) |
| Q | Glutamine | Polar |
| R | Arginine | Basic (positive charge) |
| S | Serine | Polar |
| T | Threonine | Polar |
| V | Valine | Hydrophobic |
| W | Tryptophan | Hydrophobic/Aromatic |
| Y | Tyrosine | Polar/Aromatic |
