"""
Utility functions for protforge MPNN module.
"""

from .formatting import (
    get_model_display_name,
    format_residue_ranges,
)

from .io import (
    setup_output_dirs,
    get_pdb_name,
    clean_pdb_file,
    build_native_fasta_header,
    build_designed_fasta_header,
    format_seq_by_chains,
    clean_prody_atoms,
)

from .config import (
    load_multi_config,
    load_bias_config,
    ConfigNamespace,
    get_default_config,
    load_yaml_config,
    merge_configs,
    expand_pdb_dir,
    check_output_exists,
    prepare_task_config,
    parse_residue_range,
    parse_residue_spec,
    residue_list_to_spec,
)

from .display import (
    print_config_summary,
)

from .cli import (
    create_design_argparser,
    create_score_argparser,
)

from .residue import (
    encode_residues,
    compute_position_masks,
    compute_membrane_labels,
    compute_chain_mask,
    process_symmetry,
    get_residue_lists_for_display,
)

from .constants import (
    alphabet,
    restype_1to3,
    restype_3to1,
    restype_int_to_str,
    restype_str_to_int,
    restype_name_to_atom14_names,
    element_dict,
    element_dict_rev,
    atom_order,
)

from .pdb import (
    parse_PDB,
    featurize,
    get_score,
    get_seq_rec,
    write_full_PDB,
)

__all__ = [
    # formatting
    "get_model_display_name",
    "format_residue_ranges",
    # io
    "setup_output_dirs",
    "get_pdb_name",
    "clean_pdb_file",
    "build_native_fasta_header",
    "build_designed_fasta_header",
    "format_seq_by_chains",
    "clean_prody_atoms",
    # config
    "load_multi_config",
    "load_bias_config",
    "ConfigNamespace",
    "get_default_config",
    "load_yaml_config",
    "merge_configs",
    "expand_pdb_dir",
    "check_output_exists",
    "prepare_task_config",
    "parse_residue_range",
    "parse_residue_spec",
    "residue_list_to_spec",
    # display
    "print_config_summary",
    # cli
    "create_design_argparser",
    "create_score_argparser",
    # residue
    "encode_residues",
    "compute_position_masks",
    "compute_membrane_labels",
    "compute_chain_mask",
    "process_symmetry",
    "get_residue_lists_for_display",
    # constants
    "alphabet",
    "restype_1to3",
    "restype_3to1",
    "restype_int_to_str",
    "restype_str_to_int",
    "restype_name_to_atom14_names",
    "element_dict",
    "element_dict_rev",
    "atom_order",
    # pdb
    "parse_PDB",
    "featurize",
    "get_score",
    "get_seq_rec",
    "write_full_PDB",
]
