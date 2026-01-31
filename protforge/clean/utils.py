"""CLEAN utility functions."""

import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm


def seed_everything(seed: int = 1234):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def ensure_dirs(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_fasta(fasta_path: str) -> Dict[str, str]:
    """Parse FASTA file, return {id: sequence}."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = ''.join(current_seq)

    return sequences


class ESMEmbedder:
    """ESM-1b embedding extractor using HuggingFace transformers."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        from transformers import EsmModel, EsmTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = Path(model_path)

        # Validate local path
        if not model_dir.exists():
            raise FileNotFoundError(
                f"ESM1b weights not found: {model_dir.resolve()}\n"
                f"Run: bash scripts/download_clean_weights.sh"
            )

        has_model = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
        if not has_model:
            raise FileNotFoundError(
                f"ESM1b weights not found: {model_dir.resolve()}\n"
                f"Run: bash scripts/download_clean_weights.sh"
            )

        import transformers
        transformers.logging.set_verbosity_error()

        self.tokenizer = EsmTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = EsmModel.from_pretrained(model_path, local_files_only=True).to(self.device).eval()

    @torch.no_grad()
    def embed(self, sequences: Dict[str, str], batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """Compute mean ESM embeddings (1280-dim) for sequences."""
        embeddings = {}
        ids = list(sequences.keys())
        seqs = [sequences[i] for i in ids]

        for i in tqdm(range(0, len(seqs), batch_size), desc="Computing ESM embeddings"):
            batch_ids = ids[i:i + batch_size]
            batch_seqs = seqs[i:i + batch_size]

            inputs = self.tokenizer(batch_seqs, return_tensors="pt", padding=True,
                                    truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            hidden = self.model(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            mean_emb = (hidden * mask).sum(1) / mask.sum(1)

            for j, seq_id in enumerate(batch_ids):
                embeddings[seq_id] = mean_emb[j].cpu()

        return embeddings
