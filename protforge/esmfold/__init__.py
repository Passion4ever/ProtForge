"""
Protein structure prediction using ESMFold.

Paper: Lin et al., Science 2023. https://doi.org/10.1126/science.ade2574
"""

from .model import ESMFold, parse_fasta

__all__ = ["ESMFold", "parse_fasta"]
