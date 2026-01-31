#!/usr/bin/env python
"""
MPNN CLI - Command line interface for ProteinMPNN/LigandMPNN.

Usage:
    mpnn design --pdb_path input.pdb --out_folder output/
    mpnn design -c config.yaml
    mpnn score --pdb_path input.pdb --out_folder output/
"""

import sys


def print_help():
    """Print help message."""
    print("MPNN - Protein sequence design using ProteinMPNN/LigandMPNN")
    print()
    print("Usage:")
    print("  mpnn design [options]       Run sequence design")
    print("  mpnn design -c config.yaml  Run from YAML config")
    print("  mpnn score [options]        Score existing sequences")
    print()
    print("Examples:")
    print("  mpnn design --pdb_path input.pdb --out_folder output/")
    print("  mpnn design --pdb_dir ./pdbs/ --out_folder output/ --model_type ligand_mpnn")
    print("  mpnn design -c batch_config.yaml")
    print("  mpnn score --pdb_path input.pdb --out_folder output/")
    print()
    print("For detailed options:")
    print("  mpnn design --help")
    print("  mpnn score --help")


def cli():
    """CLI entry point for mpnn command."""
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    subcommand = sys.argv[1]

    # Handle help flag at top level
    if subcommand in ["-h", "--help"]:
        print_help()
        sys.exit(0)

    # Remove subcommand from argv so argparse works correctly
    sys.argv = [f"mpnn {subcommand}"] + sys.argv[2:]

    if subcommand == "design":
        from .utils.cli import create_design_argparser
        from .run import main as run_main
        from .batch import run_batch

        parser = create_design_argparser()
        args = parser.parse_args()

        # Show help if no input specified
        if not args.config and not args.pdb_path and not args.pdb_dir:
            parser.print_help()
            sys.exit(0)

        # If config file provided, use batch mode
        if args.config:
            from pathlib import Path
            if not Path(args.config).exists():
                print(f"Error: Config file not found: {args.config}")
                sys.exit(1)
            stats = run_batch(args.config, getattr(args, 'verbose', False))
            if stats["failed"] > 0:
                sys.exit(1)
        else:
            run_main(args)

    elif subcommand == "score":
        from .utils.cli import create_score_argparser
        from .score import main as score_main

        parser = create_score_argparser()
        args = parser.parse_args()

        # Show help if no input specified
        if not args.pdb_path and not args.pdb_dir:
            parser.print_help()
            sys.exit(0)

        score_main(args)

    else:
        print(f"Unknown subcommand: {subcommand}")
        print("Available: design, score")
        sys.exit(1)


if __name__ == "__main__":
    cli()
