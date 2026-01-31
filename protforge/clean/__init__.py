"""
EC number prediction using CLEAN (Contrastive Learning enabled Enzyme ANnotation).

Paper: Yu et al., Science 2023. https://doi.org/10.1126/science.adf2465
"""

from .infer import CLEAN
from .utils import parse_fasta, ESMEmbedder
from .evaluate import parse_labels_txt, get_eval_metrics

__all__ = ["CLEAN", "parse_fasta", "ESMEmbedder", "parse_labels_txt", "get_eval_metrics"]
