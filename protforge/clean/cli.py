"""CLEAN CLI - EC number prediction for protein sequences."""

from pathlib import Path
from typing import Optional

import typer

from .infer import CLEAN
from .evaluate import parse_labels_txt, get_eval_metrics, print_eval_report

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Default paths relative to package
DEFAULT_WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights" / "clean" / "pretrained"
DEFAULT_ESM_DIR = Path(__file__).parent.parent.parent / "weights" / "clean" / "esm1b"


@app.command()
def main(
    input_file: Path = typer.Option(..., "-i", "--input", help="Input FASTA file", exists=True, dir_okay=False),
    output_file: Path = typer.Option(..., "-o", "--output", help="Output CSV file", dir_okay=False),
    device: Optional[str] = typer.Option(None, "--device", help="Device (cuda/cuda:0/cpu)"),
    model: str = typer.Option("100", "--model", help="Model variant (100 or 70)"),
    labels: Optional[Path] = typer.Option(None, "--labels", help="True labels file for evaluation", exists=True, dir_okay=False),
    no_gmm: bool = typer.Option(False, "--no-gmm", help="Disable GMM confidence (output raw distances)"),
    clean_weights: Optional[Path] = typer.Option(None, "--clean-weights", help="CLEAN weights directory"),
    esm_weights: Optional[Path] = typer.Option(None, "--esm-weights", help="ESM-1b weights directory"),
):
    """
    CLEAN - EC Number Prediction

    Examples:
        clean -i input.fasta -o results.csv
        clean -i input.fasta -o results.csv --labels labels.txt
    """
    # Resolve paths
    weights_dir = clean_weights if clean_weights else DEFAULT_WEIGHTS_DIR
    esm_path = esm_weights if esm_weights else DEFAULT_ESM_DIR
    split = f"split{model}"  # "100" -> "split100"

    typer.echo(f"Input:   {input_file}")
    typer.echo(f"Output:  {output_file}")
    typer.echo(f"Model:   {model}")
    if labels:
        typer.echo(f"Labels:  {labels}")
    typer.echo("")

    # Load model
    typer.echo("Loading models...")
    try:
        clean_model = CLEAN(
            weights_dir=str(weights_dir),
            esm_model_path=str(esm_path),
            split=split,
            device=device,
        )
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    # Run prediction
    typer.echo("Running prediction...")
    use_gmm = not no_gmm
    results = clean_model.predict_fasta(str(input_file), str(output_file), use_gmm=use_gmm)

    # Print summary
    typer.echo("")
    typer.echo(f"Predicted EC numbers for {len(results)} sequences:")
    for seq_id, predictions in list(results.items())[:5]:
        ec_str = ", ".join([f"{ec}" for ec, _ in predictions])
        typer.echo(f"  {seq_id}: {ec_str}")

    if len(results) > 5:
        typer.echo(f"  ... and {len(results) - 5} more")

    typer.echo("")
    typer.echo(f"Results saved to: {output_file}")

    # Evaluation mode
    if labels:
        typer.echo("")
        true_labels, all_ecs = parse_labels_txt(str(labels), list(results.keys()))
        metrics = get_eval_metrics(results, true_labels, all_ecs)
        print_eval_report(metrics)


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
