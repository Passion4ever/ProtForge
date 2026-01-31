"""
Model loading utilities for protforge MPNN module.
"""

import sys
import torch

from ..models import ProteinMPNN, Packer


# Model type to checkpoint argument mapping
MODEL_CHECKPOINT_MAP = {
    "protein_mpnn": "checkpoint_protein_mpnn",
    "ligand_mpnn": "checkpoint_ligand_mpnn",
    "per_residue_label_membrane_mpnn": "checkpoint_per_residue_label_membrane_mpnn",
    "global_label_membrane_mpnn": "checkpoint_global_label_membrane_mpnn",
    "soluble_mpnn": "checkpoint_soluble_mpnn",
}


def get_checkpoint_path(model_type: str, args) -> str:
    """Get checkpoint path for the specified model type."""
    if model_type not in MODEL_CHECKPOINT_MAP:
        print(f"Unknown model type: {model_type}")
        print(f"Available: {', '.join(MODEL_CHECKPOINT_MAP.keys())}")
        sys.exit(1)
    return getattr(args, MODEL_CHECKPOINT_MAP[model_type])


def load_mpnn_model(args, device):
    """
    Load MPNN model and return (model, checkpoint_path, atom_context_num).
    """
    checkpoint_path = get_checkpoint_path(args.model_type, args)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if args.model_type == "ligand_mpnn":
        atom_context_num = checkpoint["atom_context_num"]
        ligand_mpnn_use_side_chain_context = args.ligand_mpnn_use_side_chain_context
    else:
        atom_context_num = 1
        ligand_mpnn_use_side_chain_context = 0

    k_neighbors = checkpoint["num_edges"]

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=args.model_type,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint_path, atom_context_num


def load_packer_model(args, device):
    """Load side-chain packer model."""
    model_sc = Packer(
        node_features=128,
        edge_features=128,
        num_positional_embeddings=16,
        num_chain_embeddings=16,
        num_rbf=16,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        atom_context_num=16,
        lower_bound=0.0,
        upper_bound=20.0,
        top_k=32,
        dropout=0.0,
        augment_eps=0.0,
        atom37_order=False,
        device=device,
        num_mix=3,
    )

    checkpoint_sc = torch.load(args.checkpoint_path_sc, map_location=device)
    model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
    model_sc.to(device)
    model_sc.eval()

    return model_sc
