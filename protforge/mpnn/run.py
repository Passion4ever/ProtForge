import copy
import json
import os.path
import random

import numpy as np
import torch
from prody import writePDB

from .utils import (
    alphabet,
    element_dict_rev,
    featurize,
    get_score,
    get_seq_rec,
    parse_PDB,
    restype_1to3,
    restype_int_to_str,
    restype_str_to_int,
    write_full_PDB,
)
from .models import pack_side_chains
from .utils.loader import load_mpnn_model, load_packer_model
from .utils import (
    get_model_display_name,
    setup_output_dirs,
    get_pdb_name,
    clean_pdb_file,
    build_native_fasta_header,
    build_designed_fasta_header,
    print_config_summary,
    format_seq_by_chains,
    clean_prody_atoms,
    load_multi_config,
    load_bias_config,
    encode_residues,
    compute_position_masks,
    compute_membrane_labels,
    compute_chain_mask,
    process_symmetry,
    get_residue_lists_for_display,
    ConfigNamespace,
    parse_residue_spec,
    expand_pdb_dir,
)


# Helper functions moved to utils.py and model.py
def main(args) -> None:
    """
    Inference function.

    Args:
        args: Either argparse.Namespace or dict with configuration parameters
    """
    # Support both dict and argparse.Namespace
    if isinstance(args, dict):
        args = ConfigNamespace(args)

    # Initialize random seed
    seed = args.seed if args.seed else int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Setup device and directories
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_folder = setup_output_dirs(args.out_folder, args.save_stats)

    # Load models
    model, checkpoint_path, atom_context_num = load_mpnn_model(args, device)
    model_sc = load_packer_model(args, device) if args.pack_side_chains else None

    # Load PDB paths
    if args.pdb_dir:
        recursive = getattr(args, 'recursive', False)
        pdb_paths = expand_pdb_dir(args.pdb_dir, recursive)
    elif args.pdb_path_multi:
        # Deprecated: for backward compatibility
        with open(args.pdb_path_multi) as f:
            pdb_paths = list(json.load(f))
    else:
        pdb_paths = [args.pdb_path]

    # Parse residue specifications (support both old and new format)
    fixed_residues_str = args.fixed_residues or ""
    redesigned_residues_str = args.redesigned_residues or ""

    # Convert new format to old format if needed
    if fixed_residues_str and ':' in fixed_residues_str:
        fixed_residues_str = " ".join(parse_residue_spec(fixed_residues_str))
    if redesigned_residues_str and ':' in redesigned_residues_str:
        redesigned_residues_str = " ".join(parse_residue_spec(redesigned_residues_str))

    # Load per-PDB configs using helper functions
    fixed_residues_multi = load_multi_config(
        args.fixed_residues_multi, fixed_residues_str, pdb_paths)
    redesigned_residues_multi = load_multi_config(
        args.redesigned_residues_multi, redesigned_residues_str, pdb_paths)

    # Global amino acid bias
    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    if args.bias_AA:
        for item in args.bias_AA.split(","):
            aa, val = item.split(":")
            bias_AA[restype_str_to_int[aa]] = float(val)

    # Per-residue bias/omit configs
    bias_AA_per_residue_multi = load_bias_config(
        args.bias_AA_per_residue_multi, args.bias_AA_per_residue, pdb_paths)
    omit_AA_per_residue_multi = load_bias_config(
        args.omit_AA_per_residue_multi, args.omit_AA_per_residue, pdb_paths)

    # Global amino acid omit
    omit_AA = torch.tensor([aa in args.omit_AA for aa in alphabet],
                           device=device, dtype=torch.float32)

    # Parse chains filter
    parse_these_chains_only_list = args.parse_these_chains_only.split(",") if args.parse_these_chains_only else []


    # loop over PDB paths
    for pdb in pdb_paths:
        if args.verbose:
            print("Designing protein from this path:", pdb)
        fixed_residues = fixed_residues_multi[pdb]
        redesigned_residues = redesigned_residues_multi[pdb]
        parse_all_atoms_flag = args.ligand_mpnn_use_side_chain_context or (
            args.pack_side_chains and not args.repack_everything
        )
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            pdb,
            device=device,
            chains=parse_these_chains_only_list,
            parse_all_atoms=parse_all_atoms_flag,
            parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
        )
        # Save original residue names for fixed chains preservation
        original_resnames = backbone.getResnames().copy()
        original_chain_ids = backbone.getChids().copy()

        # Encode residues using shared utility
        encoded_residues, encoded_residue_dict, encoded_residue_dict_rev = encode_residues(
            protein_dict, icodes)
        chain_letters_list = list(protein_dict["chain_letters"])

        bias_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        if args.bias_AA_per_residue_multi or args.bias_AA_per_residue:
            bias_dict = bias_AA_per_residue_multi[pdb]
            for residue_name, v1 in bias_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid, v2 in v1.items():
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            bias_AA_per_residue[i1, j1] = v2

        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        if args.omit_AA_per_residue_multi or args.omit_AA_per_residue:
            omit_dict = omit_AA_per_residue_multi[pdb]
            for residue_name, v1 in omit_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid in v1:
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            omit_AA_per_residue[i1, j1] = 1.0

        # Compute position masks
        fixed_positions, redesigned_positions = compute_position_masks(
            encoded_residues, fixed_residues, redesigned_residues, device)

        # Compute membrane labels
        protein_dict["membrane_per_residue_labels"] = compute_membrane_labels(
            args, encoded_residues, fixed_positions, device)

        # Compute chain mask
        chain_mask, chains_to_design_list = compute_chain_mask(
            protein_dict, args.chains_to_design,
            fixed_residues, redesigned_residues,
            fixed_positions, redesigned_positions, device)
        protein_dict["chain_mask"] = chain_mask

        # Count residues to be redesigned/fixed
        n_redesign = int(protein_dict["chain_mask"].sum().item())
        n_fixed = int(protein_dict["chain_mask"].shape[0]) - n_redesign
        pdb_name = os.path.basename(pdb)

        # Get chain statistics
        all_chains = sorted(set(chain_letters_list))
        designed_chains = sorted(set(chains_to_design_list))
        fixed_chains = sorted([c for c in all_chains if c not in designed_chains])

        # Print configuration summary
        model_display = get_model_display_name(args.model_type, checkpoint_path)
        print_config_summary(
            pdb_name=pdb_name,
            model_display=model_display,
            designed_chains=designed_chains,
            fixed_chains=fixed_chains,
            n_redesign=n_redesign,
            n_fixed=n_fixed,
            args=args,
            redesigned_residues=redesigned_residues,
            fixed_residues=fixed_residues,
        )

        if args.verbose:
            redesigned_list, fixed_list = get_residue_lists_for_display(
                protein_dict["chain_mask"], encoded_residue_dict_rev)
            print("  Redesigned residues:", redesigned_list)
            print("  Fixed residues:", fixed_list)

        # Process symmetry using shared utility
        remapped_symmetry_residues, symmetry_weights = process_symmetry(
            args, encoded_residues, encoded_residue_dict,
            chain_letters_list, verbose=args.verbose)

        # Set other atom bfactors to 0.0
        if other_atoms:
            other_atoms.setBetas(other_atoms.getBetas() * 0.0)

        # Get PDB name without extension
        name = get_pdb_name(pdb)

        with torch.no_grad():
            # run featurize to remap R_idx and add batch dimension
            if args.verbose:
                if "Y" in list(protein_dict):
                    atom_coords = protein_dict["Y"].cpu().numpy()
                    atom_types = list(protein_dict["Y_t"].cpu().numpy())
                    atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                    number_of_atoms_parsed = np.sum(atom_mask)
                else:
                    print("No ligand atoms parsed")
                    number_of_atoms_parsed = 0
                    atom_types = ""
                    atom_coords = []
                if number_of_atoms_parsed == 0:
                    print("No ligand atoms parsed")
                elif args.model_type == "ligand_mpnn":
                    print(
                        f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                    )
                    for i, atom_type in enumerate(atom_types):
                        print(
                            f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                        )
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=args.ligand_mpnn_cutoff_for_score,
                use_atom_context=args.ligand_mpnn_use_atom_context,
                number_of_ligand_atoms=atom_context_num,
                model_type=args.model_type,
            )
            feature_dict["batch_size"] = args.batch_size
            B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
            # Count residues near ligand (for ligand_mpnn)
            n_ligand_residues = 0
            if args.model_type == "ligand_mpnn" and "mask_XY" in feature_dict:
                n_ligand_residues = int(feature_dict["mask_XY"].sum().item())
            # add additional keys to the feature dictionary
            feature_dict["temperature"] = args.temperature
            feature_dict["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue[None]
                - 1e8 * omit_AA_per_residue[None]
            )
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights

            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_XY_list = []
            for _ in range(args.number_of_batches):
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=device,
                )
                output_dict = model.sample(feature_dict)

                # compute confidence scores
                loss, loss_per_residue = get_score(
                    output_dict["S"],
                    output_dict["log_probs"],
                    feature_dict["mask"] * feature_dict["chain_mask"],
                )
                if args.model_type == "ligand_mpnn":
                    combined_mask = (
                        feature_dict["mask"]
                        * feature_dict["mask_XY"]
                        * feature_dict["chain_mask"]
                    )
                else:
                    combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]
                loss_XY, _ = get_score(
                    output_dict["S"], output_dict["log_probs"], combined_mask
                )
                # -----
                S_list.append(output_dict["S"])
                log_probs_list.append(output_dict["log_probs"])
                sampling_probs_list.append(output_dict["sampling_probs"])
                decoding_order_list.append(output_dict["decoding_order"])
                loss_list.append(loss)
                loss_per_residue_list.append(loss_per_residue)
                loss_XY_list.append(loss_XY)
            S_stack = torch.cat(S_list, 0)
            log_probs_stack = torch.cat(log_probs_list, 0)
            sampling_probs_stack = torch.cat(sampling_probs_list, 0)
            decoding_order_stack = torch.cat(decoding_order_list, 0)
            loss_stack = torch.cat(loss_list, 0)
            loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
            loss_XY_stack = torch.cat(loss_XY_list, 0)
            rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
            rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

            native_seq = "".join(restype_int_to_str[aa] for aa in feature_dict["S"][0].cpu().numpy())
            seq_out_str = format_seq_by_chains(native_seq, protein_dict["mask_c"], args.fasta_seq_separation)

            # Output paths
            output_fasta = f"{base_folder}seqs/{name}{args.file_ending}.fa"
            output_backbones = f"{base_folder}backbones/"
            output_packed = f"{base_folder}packed/"

            # Save stats if requested
            if args.save_stats:
                out_dict = {
                    "generated_sequences": S_stack.cpu(),
                    "sampling_probs": sampling_probs_stack.cpu(),
                    "log_probs": log_probs_stack.cpu(),
                    "decoding_order": decoding_order_stack.cpu(),
                    "native_sequence": feature_dict["S"][0].cpu(),
                    "mask": feature_dict["mask"][0].cpu(),
                    "chain_mask": feature_dict["chain_mask"][0].cpu(),
                    "seed": seed,
                    "temperature": args.temperature,
                }
                torch.save(out_dict, f"{base_folder}stats/{name}{args.file_ending}.pt")

            if args.pack_side_chains:
                if args.verbose:
                    print("Packing side chains...")
                feature_dict_ = featurize(
                    protein_dict, cutoff_for_score=8.0,
                    use_atom_context=args.pack_with_ligand_context,
                    number_of_ligand_atoms=16, model_type="ligand_mpnn",
                )
                sc_feature_dict = copy.deepcopy(feature_dict_)
                B = args.batch_size
                # Repeat tensors along batch dimension
                for k, v in sc_feature_dict.items():
                    if k != "S" and hasattr(v, 'repeat'):
                        sc_feature_dict[k] = v.repeat(B, *([1] * (len(v.shape) - 1)))
                X_stack_list = []
                X_m_stack_list = []
                b_factor_stack_list = []
                for _ in range(args.number_of_packs_per_design):
                    X_list = []
                    X_m_list = []
                    b_factor_list = []
                    for c in range(args.number_of_batches):
                        sc_feature_dict["S"] = S_list[c]
                        sc_dict = pack_side_chains(
                            sc_feature_dict,
                            model_sc,
                            args.sc_num_denoising_steps,
                            args.sc_num_samples,
                            args.repack_everything,
                        )
                        X_list.append(sc_dict["X"])
                        X_m_list.append(sc_dict["X_m"])
                        b_factor_list.append(sc_dict["b_factors"])

                    X_stack = torch.cat(X_list, 0)
                    X_m_stack = torch.cat(X_m_list, 0)
                    b_factor_stack = torch.cat(b_factor_list, 0)

                    X_stack_list.append(X_stack)
                    X_m_stack_list.append(X_m_stack)
                    b_factor_stack_list.append(b_factor_stack)

            with open(output_fasta, "w") as f:
                # Build and write native sequence header
                native_header = build_native_fasta_header(
                    name=name,
                    model_display=model_display,
                    designed_chains=designed_chains,
                    fixed_chains=fixed_chains,
                    n_redesign=n_redesign,
                    n_fixed=n_fixed,
                    args=args,
                    seed=seed,
                    redesigned_residues=redesigned_residues,
                    fixed_residues=fixed_residues,
                    n_ligand_residues=n_ligand_residues,
                )
                f.write(f">{native_header}\n{seq_out_str}\n")
                for ix in range(S_stack.shape[0]):
                    ix_suffix = ix
                    if not args.zero_indexed:
                        ix_suffix += 1
                    seq_rec_print = np.format_float_positional(rec_stack[ix].cpu().numpy(), unique=False, precision=4)
                    loss_np = np.format_float_positional(np.exp(-loss_stack[ix].cpu().numpy()), unique=False, precision=4)
                    loss_XY_np = np.format_float_positional(np.exp(-loss_XY_stack[ix].cpu().numpy()), unique=False, precision=4)
                    seq = "".join(restype_int_to_str[aa] for aa in S_stack[ix].cpu().numpy())

                    # Write new sequences into PDB with backbone coordinates
                    seq_prody = np.array([restype_1to3[aa] for aa in seq])[None, :].repeat(4, 1)
                    bfactor_prody = loss_per_residue_stack[ix].cpu().numpy()[None, :].repeat(4, 1)
                    # Only update residue names for designed chains, preserve original for fixed chains
                    new_resnames = seq_prody.copy()
                    for atom_idx in range(len(original_resnames)):
                        if original_chain_ids[atom_idx] not in chains_to_design_list:
                            # Fixed chain - keep original residue name
                            new_resnames[:, atom_idx] = original_resnames[atom_idx]
                    backbone.setResnames(new_resnames)
                    backbone.setBetas(
                        np.exp(-bfactor_prody)
                        * (bfactor_prody > 0.01).astype(np.float32)
                    )
                    # Write backbone PDB (clean ANISOU records)
                    backbone_out = clean_prody_atoms(backbone)
                    other_atoms_out = clean_prody_atoms(other_atoms)
                    pdb_path = f"{output_backbones}{name}_{ix_suffix}{args.file_ending}.pdb"
                    writePDB(pdb_path, backbone_out + other_atoms_out if other_atoms_out else backbone_out)
                    clean_pdb_file(pdb_path, chains_to_design_list)

                    # Write packed PDB files
                    if args.pack_side_chains:
                        for c_pack in range(args.number_of_packs_per_design):
                            packed_path = f"{output_packed}{name}_{ix_suffix}{args.packed_suffix}_{c_pack + 1}{args.file_ending}.pdb"
                            write_full_PDB(
                                packed_path,
                                X_stack_list[c_pack][ix].cpu().numpy(),
                                X_m_stack_list[c_pack][ix].cpu().numpy(),
                                b_factor_stack_list[c_pack][ix].cpu().numpy(),
                                feature_dict["R_idx_original"][0].cpu().numpy(),
                                protein_dict["chain_letters"],
                                S_stack[ix].cpu().numpy(),
                                other_atoms=other_atoms, icodes=icodes,
                                force_hetatm=args.force_hetatm,
                            )

                    # Format sequence for FASTA output
                    seq_out_str = format_seq_by_chains(seq, protein_dict["mask_c"], args.fasta_seq_separation)
                    # Build header for designed sequence
                    seq_rec_pct = float(seq_rec_print) * 100
                    seq_header = build_designed_fasta_header(
                        name=name,
                        ix_suffix=ix_suffix,
                        conf=loss_np,
                        rec=seq_rec_pct,
                        model_type=args.model_type,
                        ligand_conf=loss_XY_np if args.model_type == "ligand_mpnn" else None,
                    )
                    if ix == S_stack.shape[0] - 1:
                        # final line (no trailing newline)
                        f.write(f">{seq_header}\n{seq_out_str}")
                    else:
                        f.write(f">{seq_header}\n{seq_out_str}\n")
            # Print output summary
            print(f"→ {output_fasta}")
            if args.pack_side_chains:
                print(f"→ {base_folder}packed/")


# Argument parser at module level for CLI integration
from .utils.cli import create_design_argparser
argparser = create_design_argparser()


if __name__ == "__main__":
    args = argparser.parse_args()
    main(args)
