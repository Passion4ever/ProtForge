"""
Command-line interface definitions for protforge MPNN module.
"""

import argparse
from pathlib import Path

# Get the weights directory relative to package
_PACKAGE_DIR = Path(__file__).parent.parent.parent.parent
_WEIGHTS_DIR = _PACKAGE_DIR / "weights" / "mpnn"


class CleanHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter for cleaner help output."""

    def __init__(self, prog, indent_increment=2, max_help_position=40, width=100):
        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        # Show short option first if available
        opts = action.option_strings
        if len(opts) == 2:
            short, long = (opts[0], opts[1]) if len(opts[0]) < len(opts[1]) else (opts[1], opts[0])
            return f"{short}, {long}"
        return ', '.join(opts)


def create_design_argparser():
    """Create argument parser for the design (run.py) module."""

    description = """
Protein sequence design using ProteinMPNN/LigandMPNN.

Examples:
  mpnn design --pdb_path input.pdb -o output/
  mpnn design --pdb_dir ./pdbs/ -o output/ --model_type ligand_mpnn
  mpnn design --pdb_path input.pdb -o output/ --fixed_residues "A:1-10 B:20-30"
  mpnn design -c config.yaml
"""

    parser = argparse.ArgumentParser(
        formatter_class=CleanHelpFormatter,
        description=description,
        usage=argparse.SUPPRESS,
    )

    # === Required ===
    required = parser.add_argument_group("Input/Output")
    required.add_argument("-c", "--config", type=str, metavar="YAML",
                         help="YAML config file (ignores other args)")
    required.add_argument("--pdb_path", type=str, metavar="FILE",
                         help="Input PDB file")
    required.add_argument("--pdb_dir", type=str, metavar="DIR",
                         help="Directory with PDB files")
    required.add_argument("-o", "--out_folder", type=str, metavar="DIR",
                         help="Output directory")
    required.add_argument("--recursive", action="store_true",
                         help="Search subdirectories")

    # === Model ===
    model = parser.add_argument_group("Model")
    model.add_argument("--model_type", type=str, default="protein_mpnn",
                       metavar="TYPE",
                       help="protein_mpnn|ligand_mpnn|soluble_mpnn|*_membrane_mpnn")
    model.add_argument("--temperature", "-t", type=float, default=0.1,
                       metavar="T", help="Sampling temperature (default: 0.1)")

    # === Residue Selection ===
    residues = parser.add_argument_group("Residue Selection (format: 'Chain:residues', e.g. 'A:1-10,20 B:5-15')")
    residues.add_argument("--fixed_residues", type=str, metavar="SPEC",
                          help="Keep these residues unchanged")
    residues.add_argument("--redesigned_residues", type=str, metavar="SPEC",
                          help="Only redesign these (others fixed)")
    residues.add_argument("--chains_to_design", type=str, metavar="A,B",
                          help="Chains to redesign")

    # === Sampling ===
    sampling = parser.add_argument_group("Sampling")
    sampling.add_argument("--seed", type=int, default=0, metavar="N",
                          help="Random seed, 0=random (default: 0)")
    sampling.add_argument("--batch_size", "-b", type=int, default=1, metavar="N",
                          help="Sequences per batch (default: 1)")
    sampling.add_argument("--number_of_batches", "-n", type=int, default=1, metavar="N",
                          help="Number of batches (default: 1)")

    # === AA Constraints ===
    aa = parser.add_argument_group("Amino Acid Constraints")
    aa.add_argument("--bias_AA", type=str, metavar="SPEC",
                    help="AA bias, e.g. 'A:-1.0,C:-3.0,W:2.0'")
    aa.add_argument("--omit_AA", type=str, metavar="AAs",
                    help="Exclude AAs globally, e.g. 'CM'")
    aa.add_argument("--homo_oligomer", action="store_true",
                    help="Identical sequences for all chains")

    # === Side-chain Packing ===
    packing = parser.add_argument_group("Side-chain Packing")
    packing.add_argument("--pack_side_chains", action="store_true",
                         help="Enable side-chain packing")

    # === Other ===
    other = parser.add_argument_group("Other")
    other.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    other.add_argument("--file_ending", type=str, default="", metavar="SUF",
                       help="Output filename suffix")

    # === Hidden/Advanced (with defaults, not shown in basic help) ===
    advanced = parser.add_argument_group("Advanced (see docs/PARAMETERS.md for details)")

    # Model checkpoints (hidden)
    for name in ["protein_mpnn", "ligand_mpnn", "per_residue_label_membrane_mpnn",
                 "global_label_membrane_mpnn", "soluble_mpnn"]:
        parser.add_argument(f"--checkpoint_{name}", type=str,
                            default=str(_WEIGHTS_DIR / f"{name.replace('_mpnn', 'mpnn')}_v_48_020.pt"
                                        if "membrane" in name or name == "soluble_mpnn"
                                        else _WEIGHTS_DIR / f"{'proteinmpnn_v_48_020.pt' if name == 'protein_mpnn' else 'ligandmpnn_v_32_010_25.pt'}"),
                            help=argparse.SUPPRESS)

    # Fix checkpoint paths
    parser.set_defaults(
        checkpoint_protein_mpnn=str(_WEIGHTS_DIR / "proteinmpnn_v_48_020.pt"),
        checkpoint_ligand_mpnn=str(_WEIGHTS_DIR / "ligandmpnn_v_32_010_25.pt"),
        checkpoint_per_residue_label_membrane_mpnn=str(_WEIGHTS_DIR / "per_residue_label_membrane_mpnn_v_48_020.pt"),
        checkpoint_global_label_membrane_mpnn=str(_WEIGHTS_DIR / "global_label_membrane_mpnn_v_48_020.pt"),
        checkpoint_soluble_mpnn=str(_WEIGHTS_DIR / "solublempnn_v_48_020.pt"),
    )

    # Advanced options
    advanced.add_argument("--parse_these_chains_only", type=str, metavar="A,B",
                          help="Only parse these chains from PDB")
    advanced.add_argument("--symmetry_residues", type=str, metavar="SPEC",
                          help="e.g. 'A12,B12|A13,B13'")
    advanced.add_argument("--symmetry_weights", type=str, metavar="SPEC",
                          help="e.g. '0.5,0.5|0.5,0.5'")
    advanced.add_argument("--bias_AA_per_residue", type=str, metavar="JSON",
                          help="Per-residue bias JSON file")
    advanced.add_argument("--omit_AA_per_residue", type=str, metavar="JSON",
                          help="Per-residue omit JSON file")

    # LigandMPNN
    advanced.add_argument("--ligand_mpnn_use_atom_context", type=int, default=1,
                          metavar="0|1", help="Use ligand atoms (default: 1)")
    advanced.add_argument("--ligand_mpnn_cutoff_for_score", type=float, default=8.0,
                          metavar="A", help="Ligand cutoff in angstroms")
    advanced.add_argument("--ligand_mpnn_use_side_chain_context", type=int, default=0,
                          metavar="0|1", help="Use side chains as context")

    # Membrane
    advanced.add_argument("--transmembrane_buried", type=str, metavar="SPEC",
                          help="Buried residues for membrane model")
    advanced.add_argument("--transmembrane_interface", type=str, metavar="SPEC",
                          help="Interface residues for membrane model")
    advanced.add_argument("--global_transmembrane_label", type=int, default=0,
                          metavar="0|1", help="Global membrane label")

    # Packing details
    advanced.add_argument("--checkpoint_path_sc", type=str,
                          default=str(_WEIGHTS_DIR / "ligandmpnn_sc_v_32_002_16.pt"),
                          help=argparse.SUPPRESS)
    advanced.add_argument("--number_of_packs_per_design", type=int, default=4,
                          metavar="N", help="Packs per design")
    advanced.add_argument("--sc_num_denoising_steps", type=int, default=3,
                          help=argparse.SUPPRESS)
    advanced.add_argument("--sc_num_samples", type=int, default=16,
                          help=argparse.SUPPRESS)
    advanced.add_argument("--repack_everything", type=int, default=0,
                          help=argparse.SUPPRESS)
    advanced.add_argument("--pack_with_ligand_context", type=int, default=1,
                          help=argparse.SUPPRESS)
    advanced.add_argument("--packed_suffix", type=str, default="_packed",
                          help=argparse.SUPPRESS)

    # Other advanced
    advanced.add_argument("--fasta_seq_separation", type=str, default=":",
                          help=argparse.SUPPRESS)
    advanced.add_argument("--force_hetatm", action="store_true",
                          help="Write ligands as HETATM")
    advanced.add_argument("--zero_indexed", action="store_true",
                          help="Numbering starts at 0")
    advanced.add_argument("--save_stats", action="store_true",
                          help="Save stats to .pt file")
    advanced.add_argument("--parse_atoms_with_zero_occupancy", action="store_true",
                          help="Parse zero-occupancy atoms")

    # Deprecated/multi-file (hidden)
    parser.add_argument("--pdb_path_multi", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--fixed_residues_multi", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--redesigned_residues_multi", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--bias_AA_per_residue_multi", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--omit_AA_per_residue_multi", type=str, default="", help=argparse.SUPPRESS)

    # Set empty string defaults for optional string args
    parser.set_defaults(
        config="", pdb_path="", pdb_dir="", out_folder="",
        fixed_residues="", redesigned_residues="", chains_to_design="",
        parse_these_chains_only="", bias_AA="", omit_AA="",
        symmetry_residues="", symmetry_weights="",
        bias_AA_per_residue="", omit_AA_per_residue="",
        transmembrane_buried="", transmembrane_interface="",
    )

    return parser


def create_score_argparser():
    """Create argument parser for the score module."""

    description = """
Score protein sequences using ProteinMPNN/LigandMPNN.

Examples:
  mpnn score --pdb_path input.pdb -o output/
  mpnn score --pdb_dir ./pdbs/ -o output/
"""

    parser = argparse.ArgumentParser(
        formatter_class=CleanHelpFormatter,
        description=description,
        usage=argparse.SUPPRESS,
    )

    # === Input/Output ===
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--pdb_path", type=str, metavar="FILE",
                          help="Input PDB file")
    io_group.add_argument("--pdb_dir", type=str, metavar="DIR",
                          help="Directory with PDB files")
    io_group.add_argument("-o", "--out_folder", type=str, metavar="DIR",
                          help="Output directory")
    io_group.add_argument("--recursive", action="store_true",
                          help="Search subdirectories")

    # === Model ===
    model = parser.add_argument_group("Model")
    model.add_argument("--model_type", type=str, default="protein_mpnn",
                       metavar="TYPE",
                       help="protein_mpnn|ligand_mpnn|soluble_mpnn|*_membrane_mpnn")

    # === Scoring Options ===
    scoring = parser.add_argument_group("Scoring")
    scoring.add_argument("--use_sequence", type=int, default=1, metavar="0|1",
                         help="Use AA sequence (default: 1)")
    scoring.add_argument("--autoregressive_score", type=int, default=0, metavar="0|1",
                         help="Autoregressive scoring (default: 0)")
    scoring.add_argument("--single_aa_score", type=int, default=1, metavar="0|1",
                         help="Single AA scoring (default: 1)")

    # === Residue Selection ===
    residues = parser.add_argument_group("Residue Selection")
    residues.add_argument("--fixed_residues", type=str, metavar="SPEC",
                          help="Fixed residues, e.g. 'A:1-10,20'")
    residues.add_argument("--redesigned_residues", type=str, metavar="SPEC",
                          help="Redesigned residues")
    residues.add_argument("--chains_to_design", type=str, metavar="A,B",
                          help="Chains to score")

    # === Other ===
    other = parser.add_argument_group("Other")
    other.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    other.add_argument("--file_ending", type=str, default="", metavar="SUF",
                       help="Output filename suffix")

    # Hidden options
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--number_of_batches", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--homo_oligomer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--symmetry_residues", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--parse_these_chains_only", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--zero_indexed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parse_atoms_with_zero_occupancy", action="store_true", help=argparse.SUPPRESS)

    # Model checkpoints (hidden)
    parser.add_argument("--checkpoint_protein_mpnn", type=str,
                        default=str(_WEIGHTS_DIR / "proteinmpnn_v_48_020.pt"), help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint_ligand_mpnn", type=str,
                        default=str(_WEIGHTS_DIR / "ligandmpnn_v_32_010_25.pt"), help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint_per_residue_label_membrane_mpnn", type=str,
                        default=str(_WEIGHTS_DIR / "per_residue_label_membrane_mpnn_v_48_020.pt"), help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint_global_label_membrane_mpnn", type=str,
                        default=str(_WEIGHTS_DIR / "global_label_membrane_mpnn_v_48_020.pt"), help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint_soluble_mpnn", type=str,
                        default=str(_WEIGHTS_DIR / "solublempnn_v_48_020.pt"), help=argparse.SUPPRESS)

    # LigandMPNN (hidden)
    parser.add_argument("--ligand_mpnn_use_atom_context", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--ligand_mpnn_cutoff_for_score", type=float, default=8.0, help=argparse.SUPPRESS)
    parser.add_argument("--ligand_mpnn_use_side_chain_context", type=int, default=0, help=argparse.SUPPRESS)

    # Membrane (hidden)
    parser.add_argument("--transmembrane_buried", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--transmembrane_interface", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--global_transmembrane_label", type=int, default=0, help=argparse.SUPPRESS)

    # Deprecated (hidden)
    parser.add_argument("--pdb_path_multi", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--fixed_residues_multi", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--redesigned_residues_multi", type=str, default="", help=argparse.SUPPRESS)

    # Set defaults
    parser.set_defaults(
        pdb_path="", pdb_dir="", out_folder="",
        fixed_residues="", redesigned_residues="",
    )

    return parser
