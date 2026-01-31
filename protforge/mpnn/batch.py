"""
Batch execution module for protforge MPNN.

Usage:
    python -m protforge.mpnn.batch --config batch_config.yaml
"""

import argparse
import sys
from pathlib import Path

from .utils import (
    load_yaml_config,
    prepare_task_config,
)
from .run import main as run_design


def run_batch(config_path: str, verbose: bool = False) -> dict:
    """
    Run batch design from YAML configuration.

    Args:
        config_path: Path to YAML config file
        verbose: Print verbose output

    Returns:
        Dict with execution statistics
    """
    # Load YAML config
    yaml_config = load_yaml_config(config_path)

    # Prepare configs for all PDBs
    pdb_configs = prepare_task_config(yaml_config)

    stats = {
        "total_pdbs": len(pdb_configs),
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }

    if not pdb_configs:
        print("No PDBs to process (all skipped or none found)")
        return stats

    print(f"PDBs to process: {len(pdb_configs)}")
    print("=" * 60)

    for idx, (config, pdb_path) in enumerate(pdb_configs):
        pdb_name = Path(pdb_path).stem
        print(f"[{idx + 1}/{len(pdb_configs)}] {pdb_name}")

        try:
            # Override verbose if specified
            if verbose:
                config["verbose"] = 1

            # Run design
            run_design(config)
            stats["completed"] += 1

        except Exception as e:
            stats["failed"] += 1
            error_msg = f"{pdb_name}: {str(e)}"
            stats["errors"].append(error_msg)
            print(f"  ERROR: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total:     {stats['total_pdbs']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed:    {stats['failed']}")

    if stats["errors"]:
        print("\nErrors:")
        for err in stats["errors"]:
            print(f"  - {err}")

    return stats


def main():
    """CLI entry point for batch execution."""
    parser = argparse.ArgumentParser(
        description="Run batch MPNN design from YAML config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    stats = run_batch(args.config, args.verbose)

    # Exit with error code if any failures
    if stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
