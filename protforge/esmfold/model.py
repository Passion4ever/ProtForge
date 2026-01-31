"""ESMFold model wrapper"""

import time
import torch
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple


def parse_fasta(fasta_path: Union[str, Path]) -> List[Tuple[str, str]]:
    """Parse FASTA file, return [(name, sequence), ...]"""
    sequences = []
    current_name = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    sequences.append((current_name, "".join(current_seq)))
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_name is not None:
            sequences.append((current_name, "".join(current_seq)))

    return sequences


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


class ESMFold:
    """ESMFold protein structure prediction"""

    DEFAULT_WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights" / "esmfold"

    def __init__(
        self,
        weights_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        from transformers import EsmForProteinFolding

        if weights_dir is None:
            weights_dir = self.DEFAULT_WEIGHTS_DIR
        weights_dir = Path(weights_dir)

        has_model = (weights_dir / "model.safetensors").exists() or (weights_dir / "pytorch_model.bin").exists()
        if not has_model:
            raise FileNotFoundError(
                f"Weights not found: {weights_dir.resolve()}\n"
                f"Run: bash scripts/download_esmfold_weights.sh"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype

        dtype_name = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, str(dtype))

        print("Loading model...", end=" ", flush=True)
        start = time.time()

        self.model = EsmForProteinFolding.from_pretrained(
            str(weights_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(device)
        self.model.eval()

        print(f"done ({format_time(time.time() - start)})")
        print(f"  Device: {device} | Precision: {dtype_name}")

    def predict(
        self,
        sequence: str,
        num_recycles: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> str:
        """Predict single sequence, return PDB string"""
        if num_recycles is not None:
            self.model.config.num_recycles = num_recycles

        with torch.no_grad():
            output = self.model.infer(sequence, chunk_size=chunk_size)
            output = {
                k: v.float() if v.dtype in (torch.bfloat16, torch.float16) else v
                for k, v in output.items()
            }
            return self.model.output_to_pdb(output)[0]

    def predict_fasta(
        self,
        fasta_path: Union[str, Path],
        output_dir: Union[str, Path],
        num_recycles: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> Dict[str, Path]:
        """Predict from FASTA file"""
        fasta_path = Path(fasta_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sequences = parse_fasta(fasta_path)
        total = len(sequences)
        print(f"\nInput: {fasta_path.name} ({total} sequences)")
        print(f"Output: {output_dir}/")
        print("-" * 50)

        results = {}
        total_start = time.time()

        for i, (name, seq) in enumerate(sequences):
            print(f"[{i+1}/{total}] {name} ({len(seq)} aa) ... ", end="", flush=True)

            start = time.time()
            pdb_string = self.predict(seq, num_recycles, chunk_size)
            elapsed = time.time() - start

            out_file = output_dir / f"{name}.pdb"
            with open(out_file, "w") as f:
                f.write(pdb_string)

            results[name] = out_file
            print(f"done ({format_time(elapsed)})")

        total_time = time.time() - total_start
        print("-" * 50)
        print(f"Finished! {total} sequences in {format_time(total_time)}")
        if total > 1:
            print(f"Average: {format_time(total_time / total)}/seq")

        return results
