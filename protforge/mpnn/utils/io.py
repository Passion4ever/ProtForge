"""
I/O utilities for protforge MPNN module.
"""

import os

import numpy as np

from .formatting import format_residue_ranges


def format_seq_by_chains(seq: str, mask_c: list, separator: str = ":") -> str:
    """Format sequence string with chain separators."""
    seq_np = np.array(list(seq))
    return separator.join(''.join(seq_np[m.cpu().numpy()]) for m in mask_c)


def clean_prody_atoms(atoms):
    """Remove ANISOU data from prody atoms and return a clean copy."""
    if atoms is None:
        return None
    out = atoms.copy()
    if out.getAnisous() is not None:
        out.delData('anisou')
    return out


def setup_output_dirs(base_folder: str, save_stats: bool = False) -> str:
    """
    Create output directories for results.
    Returns normalized base_folder path (with trailing slash).
    """
    if base_folder[-1] != "/":
        base_folder = base_folder + "/"

    for subdir in ["seqs", "backbones", "packed"]:
        os.makedirs(base_folder + subdir, exist_ok=True)

    if save_stats:
        os.makedirs(base_folder + "stats", exist_ok=True)

    return base_folder


def get_pdb_name(pdb_path: str) -> str:
    """Extract PDB name from path, removing .pdb extension."""
    name = pdb_path[pdb_path.rfind("/") + 1:]
    if name.endswith(".pdb"):
        name = name[:-4]
    return name


def clean_pdb_file(pdb_path: str, designed_chains: list):
    """
    Clean up PDB file:
    - Add REMARK about B-factor meaning
    - Convert HETATM to ATOM for designed chains
    - Remove altloc identifiers for designed chains
    """
    with open(pdb_path, 'r') as f:
        pdb_lines = f.readlines()

    with open(pdb_path, 'w') as f:
        f.write("REMARK   B-factor contains design confidence (exp(-loss)), not temperature factor\n")
        for line in pdb_lines:
            if line.startswith("REMARK"):
                continue
            chain_id = line[21:22] if len(line) > 21 else ""
            if chain_id in designed_chains:
                if line.startswith("HETATM"):
                    line = "ATOM  " + line[6:]
                if line.startswith("ATOM") and len(line) > 16:
                    line = line[:16] + " " + line[17:]
            f.write(line)


def build_native_fasta_header(
    name: str,
    model_display: str,
    designed_chains: list,
    fixed_chains: list,
    n_redesign: int,
    n_fixed: int,
    args,
    seed: int,
    redesigned_residues: list = None,
    fixed_residues: list = None,
    n_ligand_residues: int = 0,
) -> str:
    """Build FASTA header for native/reference sequence."""
    header_parts = [name, model_display]

    # Chain info
    if fixed_chains:
        header_parts.append(f"design:{','.join(designed_chains)}")
        header_parts.append(f"fixed:{','.join(fixed_chains)}")
    else:
        header_parts.append(f"design:{','.join(designed_chains)}")

    # Residue constraints
    if redesigned_residues:
        header_parts.append(f"design_only:{format_residue_ranges(redesigned_residues)}")
    elif fixed_residues:
        header_parts.append(f"fix:{format_residue_ranges(fixed_residues)}")

    # Residue count
    if args.model_type == "ligand_mpnn" and n_ligand_residues > 0:
        header_parts.append(f"{n_redesign}/{n_redesign + n_fixed} res ({n_ligand_residues} near ligand)")
    else:
        header_parts.append(f"{n_redesign}/{n_redesign + n_fixed} res")

    # Ligand context info
    if args.model_type == "ligand_mpnn":
        if args.ligand_mpnn_use_atom_context:
            header_parts.append("ligand_context:on")
            header_parts.append(f"cutoff:{args.ligand_mpnn_cutoff_for_score}A")
            if args.ligand_mpnn_use_side_chain_context:
                header_parts.append("sidechain:on")
        else:
            header_parts.append("ligand_context:off")

    # Bias/Omit
    if args.bias_AA:
        bias_items = [(item.split(":")[0], float(item.split(":")[1])) for item in args.bias_AA.split(",")]
        pos_aa = ''.join([aa for aa, val in bias_items if val > 0])
        neg_aa = ''.join([aa for aa, val in bias_items if val < 0])
        bias_str = ""
        if pos_aa:
            bias_str += f"+{pos_aa}"
        if neg_aa:
            bias_str += f"-{neg_aa}"
        header_parts.append(f"bias:{bias_str}")

    if args.omit_AA:
        omit_unique = ''.join(sorted(set(args.omit_AA)))
        header_parts.append(f"omit:{omit_unique}")

    if args.homo_oligomer:
        header_parts.append("homo-oligomer")

    header_parts.append(f"T={args.temperature}")
    header_parts.append(f"seed={seed}")

    return " | ".join(header_parts)


def build_designed_fasta_header(
    name: str,
    ix_suffix: int,
    conf: str,
    rec: float,
    model_type: str,
    ligand_conf: str = None,
) -> str:
    """Build FASTA header for designed sequence."""
    header_parts = [f"{name}_{ix_suffix}", f"conf={conf}"]

    if model_type == "ligand_mpnn" and ligand_conf:
        header_parts.append(f"ligand_conf={ligand_conf}")

    header_parts.append(f"rec={rec:.1f}%")

    return " | ".join(header_parts)
