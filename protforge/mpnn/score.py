import json
import os.path
import random
import sys

import numpy as np
import torch

from .utils.loader import load_mpnn_model
from .utils import (
    alphabet,
    element_dict_rev,
    featurize,
    parse_PDB,
    restype_int_to_str,
    load_multi_config,
    get_pdb_name,
    encode_residues,
    compute_position_masks,
    compute_membrane_labels,
    compute_chain_mask,
    process_symmetry,
    get_residue_lists_for_display,
    expand_pdb_dir,
)


def main(args) -> None:
    """
    Inference function
    """
    seed = args.seed if args.seed else int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_folder = args.out_folder.rstrip("/") + "/"
    os.makedirs(base_folder, exist_ok=True)

    # Load model using shared loader
    model, checkpoint_path, atom_context_num = load_mpnn_model(args, device)

    # Load PDB paths
    if args.pdb_path_multi:
        with open(args.pdb_path_multi) as f:
            pdb_paths = list(json.load(f))
    elif args.pdb_dir:
        pdb_paths = expand_pdb_dir(args.pdb_dir, getattr(args, 'recursive', False))
    else:
        pdb_paths = [args.pdb_path]

    # Load per-PDB configs using helper functions
    fixed_residues_multi = load_multi_config(
        args.fixed_residues_multi, args.fixed_residues, pdb_paths)
    redesigned_residues_multi = load_multi_config(
        args.redesigned_residues_multi, args.redesigned_residues, pdb_paths)

    # loop over PDB paths
    for pdb in pdb_paths:
        if args.verbose:
            print("Scoring protein from this path:", pdb)

        fixed_residues = fixed_residues_multi[pdb]
        redesigned_residues = redesigned_residues_multi[pdb]

        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            pdb,
            device=device,
            chains=args.parse_these_chains_only,
            parse_all_atoms=args.ligand_mpnn_use_side_chain_context,
            parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy
        )

        # Encode residues using shared utility
        encoded_residues, encoded_residue_dict, encoded_residue_dict_rev = encode_residues(
            protein_dict, icodes)
        chain_letters_list = list(protein_dict["chain_letters"])

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

        if args.verbose:
            redesigned_list, fixed_list = get_residue_lists_for_display(
                protein_dict["chain_mask"], encoded_residue_dict_rev)
            print("These residues will be redesigned: ", redesigned_list)
            print("These residues will be fixed: ", fixed_list)

        # Process symmetry
        remapped_symmetry_residues, _ = process_symmetry(
            args, encoded_residues, encoded_residue_dict,
            chain_letters_list, verbose=args.verbose)

        # Set other atom bfactors to 0.0
        if other_atoms:
            other_atoms.setBetas(other_atoms.getBetas() * 0.0)

        # Get PDB name
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
            # add additional keys to the feature dictionary
            feature_dict["symmetry_residues"] = remapped_symmetry_residues

            logits_list = []
            probs_list = []
            log_probs_list = []
            decoding_order_list = []
            for _ in range(args.number_of_batches):
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=device,
                )
                if args.autoregressive_score:
                    score_dict = model.score(feature_dict, use_sequence=args.use_sequence)
                elif args.single_aa_score:
                    score_dict = model.single_aa_score(feature_dict, use_sequence=args.use_sequence)
                else:
                    print("Set either autoregressive_score or single_aa_score to True")
                    sys.exit()
                logits_list.append(score_dict["logits"])
                log_probs_list.append(score_dict["log_probs"])
                probs_list.append(torch.exp(score_dict["log_probs"]))
                decoding_order_list.append(score_dict["decoding_order"])
            log_probs_stack = torch.cat(log_probs_list, 0)
            logits_stack = torch.cat(logits_list, 0)
            probs_stack = torch.cat(probs_list, 0)
            decoding_order_stack = torch.cat(decoding_order_list, 0)

            output_stats_path = base_folder + name + args.file_ending + ".pt"
            out_dict = {}
            out_dict["logits"] = logits_stack.cpu().numpy()
            out_dict["probs"] = probs_stack.cpu().numpy()
            out_dict["log_probs"] = log_probs_stack.cpu().numpy()
            out_dict["decoding_order"] = decoding_order_stack.cpu().numpy()
            out_dict["native_sequence"] = feature_dict["S"][0].cpu().numpy()
            out_dict["mask"] = feature_dict["mask"][0].cpu().numpy()
            out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu().numpy() #this affects decoding order
            out_dict["seed"] = seed
            out_dict["alphabet"] = alphabet
            out_dict["residue_names"] = encoded_residue_dict_rev

            mean_probs = np.mean(out_dict["probs"], 0)
            std_probs = np.std(out_dict["probs"], 0)
            sequence = [restype_int_to_str[AA] for AA in out_dict["native_sequence"]]
            mean_dict = {}
            std_dict = {}
            for residue in range(L):
                mean_dict_ = dict(zip(alphabet, mean_probs[residue]))
                mean_dict[encoded_residue_dict_rev[residue]] = mean_dict_
                std_dict_ = dict(zip(alphabet, std_probs[residue]))
                std_dict[encoded_residue_dict_rev[residue]] = std_dict_

            out_dict["sequence"] = sequence
            out_dict["mean_of_probs"] = mean_dict
            out_dict["std_of_probs"] = std_dict
            torch.save(out_dict, output_stats_path)



# Argument parser at module level for CLI integration
from .utils.cli import create_score_argparser
argparser = create_score_argparser()


if __name__ == "__main__":
    args = argparser.parse_args()
    main(args)
