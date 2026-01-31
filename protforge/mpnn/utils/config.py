"""
Configuration loading utilities for protforge MPNN module.
"""

import json
import re
from pathlib import Path
from typing import Any, List


def load_multi_config(multi_path: str, single_value: str, pdb_paths: list, split_str: bool = True) -> dict:
    """
    Load config from multi-file JSON or broadcast single value to all pdbs.

    Args:
        multi_path: Path to JSON file with per-pdb configs
        single_value: Single value to broadcast if multi_path is empty
        pdb_paths: List of PDB paths
        split_str: If True, split string values by whitespace

    Returns:
        Dict mapping pdb_path -> config value
    """
    if multi_path:
        with open(multi_path) as f:
            data = json.load(f)
            if split_str:
                return {k: v.split() if isinstance(v, str) else v for k, v in data.items()}
            return data

    value = single_value.split() if (split_str and single_value) else (single_value or [])
    return {pdb: value for pdb in pdb_paths}


def load_bias_config(multi_path: str, single_path: str, pdb_paths: list) -> dict:
    """
    Load per-residue bias/omit config from JSON files.

    Args:
        multi_path: Path to multi-pdb JSON {pdb_path: {residue: config}}
        single_path: Path to single-pdb JSON {residue: config}
        pdb_paths: List of PDB paths

    Returns:
        Dict mapping pdb_path -> {residue: config}
    """
    if multi_path:
        with open(multi_path) as f:
            return json.load(f)

    if single_path:
        with open(single_path) as f:
            single_config = json.load(f)
            return {pdb: single_config for pdb in pdb_paths}

    return {}


# =============================================================================
# Residue Format Parsing
# =============================================================================

def parse_residue_range(range_str: str) -> List[int]:
    """
    Parse a residue range string like '1-10' or '20'.

    Args:
        range_str: Range string (e.g., '1-10', '20')

    Returns:
        List of residue numbers

    Raises:
        ValueError: If range is invalid (reversed or equal)
    """
    range_str = range_str.strip()
    if not range_str:
        return []

    if '-' in range_str:
        # Handle negative numbers: only split on '-' that's not at the start
        # e.g., "-5-10" should be parsed as -5 to 10
        parts = range_str.split('-')

        # Handle cases like "1-10", "-5", "-5-10", "5--10" (invalid)
        if range_str.startswith('-'):
            # Negative start: e.g., "-5-10" -> ['-5', '10'] or "-5" -> ['-5']
            if len(parts) == 2:
                # Just "-5"
                return [int(range_str)]
            elif len(parts) == 3:
                # "-5-10"
                start = int('-' + parts[1])
                end = int(parts[2])
            else:
                raise ValueError(f"Invalid range format: '{range_str}'")
        else:
            # Normal case: "1-10"
            if len(parts) != 2:
                raise ValueError(f"Invalid range format: '{range_str}'")
            start = int(parts[0])
            end = int(parts[1])

        if start > end:
            raise ValueError(f"Invalid range '{start}-{end}': start > end")
        if start == end:
            raise ValueError(f"Invalid range '{start}-{end}': use '{start}' instead of '{start}-{end}'")

        return list(range(start, end + 1))
    else:
        return [int(range_str)]


def parse_residue_spec(spec: str) -> List[str]:
    """
    Parse residue specification in the new format: 'A:1-10,20 B:5-15,25'

    Args:
        spec: Residue specification string

    Returns:
        List of residue identifiers (e.g., ['A1', 'A2', ..., 'A10', 'A20', 'B5', ...])

    Raises:
        ValueError: If format is invalid
    """
    if not spec or not spec.strip():
        return []

    spec = spec.strip()
    result = []

    # Check if it's the new format (contains ':') or old format
    if ':' in spec:
        # New format: 'A:1-10,20 B:5-15,25'
        # Split by whitespace to get chain specifications
        chain_specs = spec.split()

        for chain_spec in chain_specs:
            if ':' not in chain_spec:
                raise ValueError(f"Invalid chain specification: '{chain_spec}'. Expected format: 'A:1-10,20'")

            chain, residues_str = chain_spec.split(':', 1)
            chain = chain.strip()

            if not chain:
                raise ValueError(f"Empty chain ID in: '{chain_spec}'")
            if len(chain) != 1:
                raise ValueError(f"Chain ID must be single character, got: '{chain}'")

            # Parse residue numbers
            residue_parts = residues_str.split(',')
            for part in residue_parts:
                part = part.strip()
                if not part:
                    continue
                try:
                    residue_nums = parse_residue_range(part)
                    for num in residue_nums:
                        result.append(f"{chain}{num}")
                except ValueError as e:
                    raise ValueError(f"Error parsing residues for chain {chain}: {e}")
    else:
        # Old format: 'A12 A13 B25' - pass through as list
        result = spec.split()

    return result


def residue_list_to_spec(residues: List[str]) -> str:
    """
    Convert a list of residue identifiers back to spec format.

    Args:
        residues: List like ['A1', 'A2', 'A3', 'B5', 'B6']

    Returns:
        Spec string like 'A:1-3 B:5-6'
    """
    if not residues:
        return ""

    # Group by chain
    chains = {}
    for res in residues:
        if not res:
            continue
        chain = res[0]
        num = int(res[1:])
        if chain not in chains:
            chains[chain] = []
        chains[chain].append(num)

    # Sort and format each chain
    parts = []
    for chain in sorted(chains.keys()):
        nums = sorted(chains[chain])
        # Compress into ranges
        ranges = []
        start = nums[0]
        end = nums[0]
        for n in nums[1:]:
            if n == end + 1:
                end = n
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = n
                end = n
        # Don't forget the last range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")

        parts.append(f"{chain}:{','.join(ranges)}")

    return " ".join(parts)


# =============================================================================
# YAML Batch Configuration Support
# =============================================================================

class ConfigNamespace:
    """Convert dict to argparse-like namespace object."""

    def __init__(self, config_dict: dict):
        self.__dict__.update(config_dict)

    def __getattr__(self, name: str) -> Any:
        # Return None for missing attributes (like argparse defaults)
        return None

    def __repr__(self) -> str:
        return f"ConfigNamespace({self.__dict__})"


def get_default_config() -> dict:
    """
    Get default configuration values matching cli.py argparser defaults.

    Returns:
        Dict with all default parameter values
    """
    # Get weights directory
    _PACKAGE_DIR = Path(__file__).parent.parent.parent.parent
    _WEIGHTS_DIR = _PACKAGE_DIR / "weights" / "mpnn"

    return {
        # Model selection
        "model_type": "protein_mpnn",
        "checkpoint_protein_mpnn": str(_WEIGHTS_DIR / "proteinmpnn_v_48_020.pt"),
        "checkpoint_ligand_mpnn": str(_WEIGHTS_DIR / "ligandmpnn_v_32_010_25.pt"),
        "checkpoint_per_residue_label_membrane_mpnn": str(_WEIGHTS_DIR / "per_residue_label_membrane_mpnn_v_48_020.pt"),
        "checkpoint_global_label_membrane_mpnn": str(_WEIGHTS_DIR / "global_label_membrane_mpnn_v_48_020.pt"),
        "checkpoint_soluble_mpnn": str(_WEIGHTS_DIR / "solublempnn_v_48_020.pt"),

        # Input/Output
        "pdb_path": "",
        "pdb_dir": "",
        "recursive": False,
        "pdb_path_multi": "",
        "out_folder": "",
        "file_ending": "",
        "fasta_seq_separation": ":",

        # Residue selection
        "fixed_residues": "",
        "fixed_residues_multi": "",
        "redesigned_residues": "",
        "redesigned_residues_multi": "",
        "chains_to_design": "",
        "parse_these_chains_only": "",

        # Amino acid bias/omit
        "bias_AA": "",
        "bias_AA_per_residue": "",
        "bias_AA_per_residue_multi": "",
        "omit_AA": "",
        "omit_AA_per_residue": "",
        "omit_AA_per_residue_multi": "",

        # Symmetry
        "symmetry_residues": "",
        "symmetry_weights": "",
        "homo_oligomer": False,

        # Sampling parameters
        "seed": 0,
        "batch_size": 1,
        "number_of_batches": 1,
        "temperature": 0.1,

        # LigandMPNN specific
        "ligand_mpnn_use_atom_context": 1,
        "ligand_mpnn_cutoff_for_score": 8.0,
        "ligand_mpnn_use_side_chain_context": 0,

        # Membrane model specific
        "transmembrane_buried": "",
        "transmembrane_interface": "",
        "global_transmembrane_label": 0,

        # Side-chain packing
        "pack_side_chains": False,
        "checkpoint_path_sc": str(_WEIGHTS_DIR / "ligandmpnn_sc_v_32_002_16.pt"),
        "number_of_packs_per_design": 4,
        "sc_num_denoising_steps": 3,
        "sc_num_samples": 16,
        "repack_everything": 0,
        "pack_with_ligand_context": 1,
        "packed_suffix": "_packed",
        "force_hetatm": False,

        # Other
        "verbose": False,
        "zero_indexed": False,
        "save_stats": False,
        "parse_atoms_with_zero_occupancy": False,
    }


def load_yaml_config(yaml_path: str) -> dict:
    """
    Load batch configuration from YAML file.

    Args:
        yaml_path: Path to YAML config file

    Returns:
        Dict with task configuration (flat format)
    """
    import yaml

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # Validate required keys
    if not config.get("pdb_dir") and not config.get("pdb_path"):
        raise ValueError("YAML config must contain 'pdb_dir' or 'pdb_path'")

    return config


def merge_configs(defaults: dict, yaml_defaults: dict, task_config: dict) -> dict:
    """
    Merge configuration with priority: task > yaml_defaults > code_defaults.

    Args:
        defaults: Code-level default values
        yaml_defaults: YAML file's defaults section
        task_config: Individual task configuration

    Returns:
        Merged configuration dict
    """
    merged = defaults.copy()
    merged.update(yaml_defaults)
    merged.update(task_config)
    return merged


def expand_pdb_dir(pdb_dir: str, recursive: bool = False) -> list:
    """
    Expand pdb_dir to list of PDB file paths.

    Args:
        pdb_dir: Directory containing PDB files
        recursive: If True, search subdirectories

    Returns:
        List of PDB file paths (sorted)
    """
    pdb_dir = Path(pdb_dir)

    if not pdb_dir.exists():
        raise ValueError(f"PDB directory does not exist: {pdb_dir}")

    if recursive:
        pdb_files = list(pdb_dir.rglob("*.pdb"))
    else:
        pdb_files = list(pdb_dir.glob("*.pdb"))

    if not pdb_files:
        raise ValueError(f"No PDB files found in: {pdb_dir}")

    return sorted([str(p) for p in pdb_files])


def check_output_exists(out_folder: str, pdb_path: str, file_ending: str = "") -> bool:
    """
    Check if output files already exist for a given PDB.

    Args:
        out_folder: Output folder path
        pdb_path: Input PDB path
        file_ending: File ending suffix

    Returns:
        True if output FASTA exists
    """
    name = Path(pdb_path).stem
    output_fasta = Path(out_folder) / "seqs" / f"{name}{file_ending}.fa"
    return output_fasta.exists()


def prepare_task_config(yaml_config: dict) -> list:
    """
    Prepare task configuration for execution.

    Handles:
    - Merging with code defaults
    - Converting residue formats
    - Expanding pdb_dir to pdb_path list
    - Setting up output folder

    Args:
        yaml_config: Flat YAML configuration dict

    Returns:
        List of (config_dict, pdb_path) tuples ready for execution
    """
    # Get code defaults
    code_defaults = get_default_config()

    # Merge: code defaults <- yaml config
    config = code_defaults.copy()
    config.update(yaml_config)

    # Extract batch-specific keys (not passed to run.py)
    pdb_dir = config.pop("pdb_dir", None)
    recursive = config.pop("recursive", False)
    skip_existing = config.pop("skip_existing", False)

    # Validate output folder
    if not config.get("out_folder"):
        raise ValueError("YAML config must contain 'out_folder'")

    # Handle external file references
    for key in ["bias_AA_per_residue_file", "omit_AA_per_residue_file"]:
        if key in config:
            # Map to the correct parameter name
            target_key = key.replace("_file", "")
            config[target_key] = config.pop(key)

    # Expand PDB paths
    if pdb_dir:
        pdb_paths = expand_pdb_dir(pdb_dir, recursive)
    elif config.get("pdb_path"):
        pdb_paths = [config["pdb_path"]]
    else:
        raise ValueError("Config must specify 'pdb_dir' or 'pdb_path'")

    # Generate configs for each PDB
    results = []
    for pdb_path in pdb_paths:
        # Check skip_existing
        if skip_existing and check_output_exists(
            config["out_folder"], pdb_path, config.get("file_ending", "")
        ):
            continue

        pdb_config = config.copy()
        pdb_config["pdb_path"] = pdb_path
        results.append((pdb_config, pdb_path))

    return results
