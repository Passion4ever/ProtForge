"""
Formatting utilities for protforge MPNN module.
"""

import re
from pathlib import Path


def get_model_display_name(model_type: str, checkpoint_path: str) -> str:
    """
    Get simplified model name with version from checkpoint path.
    Examples:
        protein_mpnn + proteinmpnn_v_48_020.pt → protein_mpnn (v48_020)
        ligand_mpnn + ligandmpnn_v_32_010_25.pt → ligand_mpnn (v32_010_25)
        per_residue_label_membrane_mpnn → residue_membrane (v48_020)
        global_label_membrane_mpnn → global_membrane (v48_020)
    """
    name_map = {
        "protein_mpnn": "protein_mpnn",
        "ligand_mpnn": "ligand_mpnn",
        "per_residue_label_membrane_mpnn": "residue_membrane",
        "global_label_membrane_mpnn": "global_membrane",
        "soluble_mpnn": "soluble_mpnn",
    }
    display_name = name_map.get(model_type, model_type)

    # Extract version from checkpoint filename
    version = ""
    if checkpoint_path:
        filename = Path(checkpoint_path).stem
        match = re.search(r'_v_(\d+_\d+(?:_\d+)?)', filename)
        if match:
            version = "v" + match.group(1)

    if version:
        return f"{display_name} ({version})"
    return display_name


def format_residue_ranges(residues: list) -> str:
    """
    Format residue list into compact range notation.
    Input: ['A78', 'A79', 'A80', 'A81', 'A82', 'A85', 'B10', 'B15', 'B16']
    Output: 'A(78-82,85) B(10,15-16)'
    """
    if not residues:
        return ""

    # Group by chain
    chain_residues = {}
    for res in residues:
        match = re.match(r'([A-Za-z]+)(\d+)(.*)', res)
        if match:
            chain = match.group(1)
            resnum = int(match.group(2))
            icode = match.group(3)
            if chain not in chain_residues:
                chain_residues[chain] = []
            chain_residues[chain].append((resnum, icode))

    # Format each chain
    parts = []
    for chain in sorted(chain_residues.keys()):
        nums = chain_residues[chain]
        nums.sort(key=lambda x: (x[0], x[1]))

        # Convert to ranges
        ranges = []
        i = 0
        while i < len(nums):
            start_num, start_icode = nums[i]
            end_num, end_icode = nums[i]

            if not start_icode:
                while (i + 1 < len(nums) and
                       not nums[i+1][1] and
                       nums[i+1][0] == end_num + 1):
                    i += 1
                    end_num = nums[i][0]

            if start_num == end_num:
                ranges.append(f"{start_num}{start_icode}")
            else:
                ranges.append(f"{start_num}-{end_num}")
            i += 1

        parts.append(f"{chain}({','.join(ranges)})")

    return ' '.join(parts)
