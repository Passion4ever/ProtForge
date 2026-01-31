"""
Residue encoding and processing utilities for protforge MPNN module.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional

from .config import parse_residue_spec


def encode_residues(protein_dict: dict, icodes: list) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Create residue encoding mappings from protein dict.

    Args:
        protein_dict: Parsed protein dictionary with R_idx and chain_letters
        icodes: Insertion codes list

    Returns:
        Tuple of (encoded_residues list, name->idx dict, idx->name dict)
    """
    R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
    chain_letters_list = list(protein_dict["chain_letters"])

    encoded_residues = [
        f"{chain_letters_list[i]}{R_idx_list[i]}{icodes[i]}"
        for i in range(len(R_idx_list))
    ]

    encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
    encoded_residue_dict_rev = dict(zip(range(len(encoded_residues)), encoded_residues))

    return encoded_residues, encoded_residue_dict, encoded_residue_dict_rev


def compute_position_masks(
    encoded_residues: List[str],
    fixed_residues: List[str],
    redesigned_residues: List[str],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute fixed and redesigned position masks.

    Args:
        encoded_residues: List of encoded residue names
        fixed_residues: List of residues to fix
        redesigned_residues: List of residues to redesign
        device: Torch device

    Returns:
        Tuple of (fixed_positions, redesigned_positions) tensors
    """
    fixed_set = set(fixed_residues)
    redesigned_set = set(redesigned_residues)

    fixed_positions = torch.tensor(
        [int(item not in fixed_set) for item in encoded_residues],
        device=device,
    )
    redesigned_positions = torch.tensor(
        [int(item not in redesigned_set) for item in encoded_residues],
        device=device,
    )

    return fixed_positions, redesigned_positions


def compute_membrane_labels(
    args,
    encoded_residues: List[str],
    fixed_positions: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute membrane per-residue labels for membrane MPNN models.

    Args:
        args: Command line arguments with transmembrane_buried, transmembrane_interface,
              model_type, global_transmembrane_label
        encoded_residues: List of encoded residue names
        fixed_positions: Fixed positions tensor (used for shape reference)
        device: Torch device

    Returns:
        Membrane per-residue labels tensor
    """
    # Buried positions
    if args.transmembrane_buried:
        # Support both old format "A12 A13" and new format "A:12,13"
        buried_spec = args.transmembrane_buried
        buried_residues = set(parse_residue_spec(buried_spec))
        buried_positions = torch.tensor(
            [int(item in buried_residues) for item in encoded_residues],
            device=device,
        )
    else:
        buried_positions = torch.zeros_like(fixed_positions)

    # Interface positions
    if args.transmembrane_interface:
        # Support both old format "A12 A13" and new format "A:12,13"
        interface_spec = args.transmembrane_interface
        interface_residues = set(parse_residue_spec(interface_spec))
        interface_positions = torch.tensor(
            [int(item in interface_residues) for item in encoded_residues],
            device=device,
        )
    else:
        interface_positions = torch.zeros_like(fixed_positions)

    # Compute labels: 2 for buried, 1 for interface, 0 otherwise
    membrane_labels = (
        2 * buried_positions * (1 - interface_positions) +
        1 * interface_positions * (1 - buried_positions)
    )

    # Override for global membrane model
    if args.model_type == "global_label_membrane_mpnn":
        membrane_labels = args.global_transmembrane_label + 0 * fixed_positions

    return membrane_labels


def compute_chain_mask(
    protein_dict: dict,
    chains_to_design: Optional[str],
    fixed_residues: List[str],
    redesigned_residues: List[str],
    fixed_positions: torch.Tensor,
    redesigned_positions: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute chain mask for design.

    Args:
        protein_dict: Parsed protein dictionary
        chains_to_design: Comma-separated chain letters or None/empty for all
        fixed_residues: List of fixed residues
        redesigned_residues: List of redesigned residues
        fixed_positions: Fixed positions tensor
        redesigned_positions: Redesigned positions tensor
        device: Torch device

    Returns:
        Tuple of (chain_mask tensor, chains_to_design_list)
    """
    # Parse chains to design
    if chains_to_design and isinstance(chains_to_design, str) and len(chains_to_design) > 0:
        chains_to_design_list = chains_to_design.split(",")
    else:
        chains_to_design_list = list(protein_dict["chain_letters"])

    # Base chain mask
    chain_mask = torch.tensor(
        np.array(
            [item in chains_to_design_list for item in protein_dict["chain_letters"]],
            dtype=np.int32,
        ),
        device=device,
    )

    # Apply fixed/redesigned constraints
    if redesigned_residues:
        chain_mask = chain_mask * (1 - redesigned_positions)
    elif fixed_residues:
        chain_mask = chain_mask * fixed_positions

    return chain_mask, chains_to_design_list


def process_symmetry(
    args,
    encoded_residues: List[str],
    encoded_residue_dict: Dict[str, int],
    chain_letters_list: List[str],
    verbose: bool = False
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Process symmetry residues and weights for homo-oligomer or custom symmetry.

    Args:
        args: Command line arguments with symmetry_residues, symmetry_weights, homo_oligomer
        encoded_residues: List of encoded residue names
        encoded_residue_dict: Mapping from residue name to index
        chain_letters_list: List of chain letters
        verbose: Whether to print verbose output

    Returns:
        Tuple of (remapped_symmetry_residues, symmetry_weights)
    """
    # Default empty
    remapped_symmetry_residues = [[]]
    symmetry_weights = [[]]

    # Custom symmetry residues
    if hasattr(args, 'symmetry_residues') and args.symmetry_residues:
        symmetry_residues_list_of_lists = [
            x.split(",") for x in args.symmetry_residues.split("|")
        ]
        remapped_symmetry_residues = [
            [encoded_residue_dict[t] for t in t_list]
            for t_list in symmetry_residues_list_of_lists
        ]

    # Custom symmetry weights
    if hasattr(args, 'symmetry_weights') and args.symmetry_weights:
        symmetry_weights = [
            [float(item) for item in x.split(",")]
            for x in args.symmetry_weights.split("|")
        ]

    # Homo-oligomer overrides custom symmetry
    if args.homo_oligomer:
        if verbose:
            print("Designing HOMO-OLIGOMER")

        chain_letters_set = list(set(chain_letters_list))
        reference_chain = chain_letters_set[0]
        lc = len(reference_chain)

        residue_indices = [
            item[lc:] for item in encoded_residues if item[:lc] == reference_chain
        ]

        remapped_symmetry_residues = []
        symmetry_weights = []
        weight = 1 / len(chain_letters_set)

        for res in residue_indices:
            tmp_list = []
            tmp_w_list = []
            for chain in chain_letters_set:
                name = chain + res
                tmp_list.append(encoded_residue_dict[name])
                tmp_w_list.append(weight)
            remapped_symmetry_residues.append(tmp_list)
            symmetry_weights.append(tmp_w_list)

    return remapped_symmetry_residues, symmetry_weights


def get_residue_lists_for_display(
    chain_mask: torch.Tensor,
    encoded_residue_dict_rev: Dict[int, str]
) -> Tuple[List[str], List[str]]:
    """
    Get lists of residues to be redesigned and fixed for display.

    Args:
        chain_mask: Chain mask tensor (1=design, 0=fixed)
        encoded_residue_dict_rev: Mapping from index to residue name

    Returns:
        Tuple of (redesigned_residues, fixed_residues) name lists
    """
    redesigned = [
        encoded_residue_dict_rev[i]
        for i in range(chain_mask.shape[0])
        if chain_mask[i] == 1
    ]
    fixed = [
        encoded_residue_dict_rev[i]
        for i in range(chain_mask.shape[0])
        if chain_mask[i] == 0
    ]
    return redesigned, fixed
