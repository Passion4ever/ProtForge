"""CLEAN inference."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .model import LayerNormNet
from .utils import ESMEmbedder, parse_fasta, seed_everything, ensure_dirs

DOWNLOAD_HINT = "Run: bash scripts/download_clean_weights.sh"


def _compute_distances(emb: torch.Tensor, centers: Dict[str, torch.Tensor],
                       device: str, dtype: torch.dtype) -> Dict[str, float]:
    """Compute L2 distances from embedding to all EC centers."""
    ec_list = list(centers.keys())
    center_tensor = torch.stack([centers[ec] for ec in ec_list]).to(device=device, dtype=dtype)
    dists = (emb.unsqueeze(0) - center_tensor).norm(dim=1, p=2).cpu().numpy()
    return {ec: float(dists[i]) for i, ec in enumerate(ec_list)}


def _max_separation(dist_lst: np.ndarray) -> int:
    """Compute max separation index for EC prediction."""
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1] - sep_lst[1:])

    large_grads = np.where(sep_grad > np.mean(sep_grad))[0]
    max_sep_i = large_grads[0] if len(large_grads) > 0 else 0
    return 0 if max_sep_i >= 5 else int(max_sep_i)


def _gmm_confidence(distance: float, gmm_lst: list) -> float:
    """Compute GMM-based confidence score."""
    conf = []
    for gmm in gmm_lst:
        a, b = gmm.means_
        idx = 0 if a[0] < b[0] else 1
        conf.append(gmm.predict_proba([[distance]])[0][idx])
    return float(np.mean(conf))


class CLEAN:
    """CLEAN: Contrastive Learning enabled Enzyme ANnotation."""

    def __init__(self, weights_dir: str, esm_model_path: str,
                 split: str = "split100", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.weights_dir = Path(weights_dir)

        # Check required files exist
        split_num = split.replace("split", "")
        checkpoint_path = self.weights_dir / f"{split}.pth"
        centers_path = self.weights_dir / f"cluster_centers_{split_num}.pt"

        if not checkpoint_path.exists() or not centers_path.exists():
            raise FileNotFoundError(f"CLEAN weights not found: {self.weights_dir.resolve()}\n{DOWNLOAD_HINT}")

        # Load CLEAN model
        self.model = LayerNormNet(512, 128, self.device, self.dtype)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        self.cluster_centers = torch.load(centers_path, map_location="cpu")
        print(f"Loaded {len(self.cluster_centers)} EC cluster centers ({split})")

        # Load ESM embedder
        self.esm = ESMEmbedder(model_path=esm_model_path, device=self.device)

        # Load GMM
        self.gmm = None
        gmm_path = self.weights_dir / "gmm_ensumble.pkl"
        if gmm_path.exists():
            try:
                self.gmm = pickle.load(open(gmm_path, 'rb'))
                print(f"Loaded GMM ensemble ({len(self.gmm)} models)")
            except Exception as e:
                print(f"Warning: Failed to load GMM: {e}")

    def predict(self, sequences: Dict[str, str], output_path: Optional[str] = None,
                use_gmm: bool = True) -> Dict[str, List[Tuple[str, float]]]:
        """Predict EC numbers for sequences."""
        seed_everything()

        # Compute embeddings
        esm_embs = self.esm.embed(sequences)
        ids = list(sequences.keys())
        esm_tensor = torch.stack([esm_embs[i] for i in ids]).to(self.device, self.dtype)

        with torch.no_grad():
            clean_embs = self.model(esm_tensor)

        # Predict for each sequence
        results = {}
        gmm = self.gmm if use_gmm else None

        for i, seq_id in enumerate(tqdm(ids, desc="Predicting EC")):
            dists = _compute_distances(clean_embs[i], self.cluster_centers, self.device, self.dtype)
            sorted_ecs = sorted(dists.items(), key=lambda x: x[1])[:10]
            dist_lst = np.array([d for _, d in sorted_ecs])

            max_sep = _max_separation(dist_lst)
            preds = []
            for j in range(max_sep + 1):
                ec, dist = sorted_ecs[j]
                score = _gmm_confidence(dist, gmm) if gmm else dist
                preds.append((ec, score))
            results[seq_id] = preds

        if output_path:
            self._save(results, output_path)
        return results

    def predict_fasta(self, fasta_path: str, output_path: Optional[str] = None,
                      use_gmm: bool = True) -> Dict[str, List[Tuple[str, float]]]:
        """Predict EC numbers from FASTA file."""
        return self.predict(parse_fasta(fasta_path), output_path, use_gmm)

    def _save(self, results: Dict[str, List[Tuple[str, float]]], path: str):
        """Save results to CSV."""
        ensure_dirs(str(Path(path).parent) or ".")
        with open(path, 'w') as f:
            for seq_id, preds in results.items():
                ec_strs = [f"EC:{ec}/{score:.4f}" for ec, score in preds]
                f.write(",".join([seq_id] + ec_strs) + "\n")
