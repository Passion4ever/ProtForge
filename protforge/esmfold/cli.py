#!/usr/bin/env python
"""ESMFold CLI"""

from pathlib import Path
from typing import Optional
import typer

app = typer.Typer(
    name="esmfold",
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def check_dtype_support(device: str, dtype_name: str) -> tuple[bool, str]:
    import torch

    if "cpu" in device:
        return False, f"CPU doesn't support {dtype_name}"

    if not torch.cuda.is_available():
        return False, "CUDA not available"

    device_idx = 0 if device == "cuda" else int(device.split(":")[1])
    if device_idx >= torch.cuda.device_count():
        return False, f"GPU {device_idx} not found"

    cap = torch.cuda.get_device_capability(device_idx)
    compute = cap[0] * 10 + cap[1]

    if dtype_name == "bf16" and compute < 80:
        return False, "bf16 requires Ampere+ GPU"
    if dtype_name == "fp16" and compute < 60:
        return False, "GPU doesn't support fp16"

    return True, ""


@app.command()
def main(
    input: Path = typer.Option(..., "-i", "--input", help="Input FASTA file", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "-o", "--output", help="Output directory", file_okay=False),
    device: Optional[str] = typer.Option(None, "--device", help="Device (cuda/cuda:0/cpu)"),
    fp16: bool = typer.Option(False, "--fp16", help="Use float16 precision"),
    bf16: bool = typer.Option(False, "--bf16", help="Use bfloat16 precision (Ampere+ required)"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Chunk size for long sequences"),
    num_recycles: Optional[int] = typer.Option(None, "--num-recycles", help="Number of recycles (default: 4)"),
    weights: Optional[Path] = typer.Option(None, "--weights", help="Weights directory"),
):
    """
    ESMFold - Protein Structure Prediction

    Examples:
        esmfold -i proteins.fasta -o results/
        esmfold -i proteins.fasta -o results/ --fp16
        esmfold -i long.fasta -o out/ --fp16 --chunk-size 64
    """
    import torch

    if fp16 and bf16:
        typer.echo("Error: --fp16 and --bf16 cannot be used together", err=True)
        raise typer.Exit(1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if bf16:
        dtype_name, dtype = "bf16", torch.bfloat16
    elif fp16:
        dtype_name, dtype = "fp16", torch.float16
    else:
        dtype_name, dtype = "fp32", torch.float32

    if dtype_name != "fp32":
        ok, reason = check_dtype_support(device, dtype_name)
        if not ok:
            typer.echo(f"Warning: {reason}, using fp32")
            dtype = torch.float32

    from .model import ESMFold

    try:
        model = ESMFold(weights_dir=weights, device=device, dtype=dtype)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    model.predict_fasta(
        fasta_path=input,
        output_dir=output,
        num_recycles=num_recycles,
        chunk_size=chunk_size,
    )


def cli():
    app()


if __name__ == "__main__":
    app()
