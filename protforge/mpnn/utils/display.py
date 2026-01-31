"""
Display utilities for protforge MPNN module.
"""

from .formatting import format_residue_ranges


def print_config_summary(
    pdb_name: str,
    model_display: str,
    designed_chains: list,
    fixed_chains: list,
    n_redesign: int,
    n_fixed: int,
    args,
    redesigned_residues: list = None,
    fixed_residues: list = None,
):
    """Print configuration summary box to terminal."""
    n_seqs = args.batch_size * args.number_of_batches

    print("╔══ PROTFORGE ══════════════════════════════════════════╗")
    print(f"║ Input    : {pdb_name}")
    print(f"║ Model    : {model_display}")

    # Chain info
    if fixed_chains:
        chain_info = f"{','.join(designed_chains)} (design) | {','.join(fixed_chains)} (fixed)"
    else:
        chain_info = f"{','.join(designed_chains)} (design)"
    if args.homo_oligomer:
        chain_info += " [homo-oligomer]"
    print(f"║ Chains   : {chain_info}")

    # Residue constraints
    if redesigned_residues:
        print(f"║ Design   : {format_residue_ranges(redesigned_residues)}")
    elif fixed_residues:
        print(f"║ Fix      : {format_residue_ranges(fixed_residues)}")
    print(f"║ Residues : {n_redesign} design / {n_fixed} fixed")

    # Ligand context settings
    if args.model_type == "ligand_mpnn":
        if args.ligand_mpnn_use_atom_context:
            context_str = f"context=on, cutoff={args.ligand_mpnn_cutoff_for_score}A"
            if args.ligand_mpnn_use_side_chain_context:
                context_str += ", sidechain=on"
        else:
            context_str = "context=off"
        print(f"║ Ligand   : {context_str}")

    # Amino acid bias/omit settings
    if args.bias_AA:
        bias_items = [(item.split(":")[0], float(item.split(":")[1])) for item in args.bias_AA.split(",")]
        value_groups = {}
        for aa, val in bias_items:
            if val not in value_groups:
                value_groups[val] = []
            value_groups[val].append(aa)
        bias_parts = []
        for val in sorted(value_groups.keys(), reverse=True):
            aas = ''.join(value_groups[val])
            sign = '+' if val > 0 else ''
            val_str = f"{sign}{val:g}"
            bias_parts.append(f"{aas}({val_str})")
        print(f"║ Bias     : {' '.join(bias_parts)}")

    if args.omit_AA:
        omit_unique = ''.join(sorted(set(args.omit_AA)))
        print(f"║ Omit     : {omit_unique}")

    print(f"║ Sampling : T={args.temperature}, {args.batch_size}×{args.number_of_batches}={n_seqs} seqs")

    if args.pack_side_chains:
        print(f"║ Packing  : {args.number_of_packs_per_design} packs/seq")

    print("╚═══════════════════════════════════════════════════════╝")
