"""CLEAN evaluation functions."""

from typing import Dict, List, Set, Tuple


def parse_labels_txt(
    labels_path: str,
    seq_ids: List[str],
) -> Tuple[Dict[str, List[str]], Set[str]]:
    """
    Parse true labels from text file.

    Args:
        labels_path: Path to labels file (one EC per line, matching FASTA order).
                     Multiple ECs separated by semicolon.
        seq_ids: List of sequence IDs (from FASTA, in order).

    Returns:
        Tuple of (true_labels, all_ecs):
        - true_labels: {seq_id: [ec_numbers]}
        - all_ecs: Set of all unique EC numbers
    """
    true_labels = {}
    all_ecs = set()

    with open(labels_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) != len(seq_ids):
        raise ValueError(
            f"Number of labels ({len(lines)}) does not match "
            f"number of sequences ({len(seq_ids)})"
        )

    for seq_id, line in zip(seq_ids, lines):
        ec_numbers = line.split(';')
        true_labels[seq_id] = ec_numbers
        all_ecs.update(ec_numbers)

    return true_labels, all_ecs


def get_eval_metrics(
    predictions: Dict[str, List[Tuple[str, float]]],
    true_labels: Dict[str, List[str]],
    all_ecs: Set[str],
) -> Dict[str, float]:
    """
    Compute evaluation metrics for EC predictions.

    Args:
        predictions: {protein_id: [(ec, confidence), ...]}
        true_labels: {protein_id: [true_ec, ...]}
        all_ecs: Set of all possible EC numbers

    Returns:
        Dict with metrics: precision, recall, f1, accuracy
    """
    try:
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            accuracy_score,
        )
    except ImportError:
        raise ImportError(
            "sklearn is required for evaluation. "
            "Install with: pip install scikit-learn"
        )

    # Get common protein IDs
    common_ids = set(predictions.keys()) & set(true_labels.keys())
    if not common_ids:
        raise ValueError("No common protein IDs between predictions and true labels")

    # Sort IDs for consistent ordering
    sorted_ids = sorted(common_ids)

    # Extract predicted and true EC lists
    pred_ec_lists = []
    true_ec_lists = []

    for prot_id in sorted_ids:
        pred_ecs = [ec for ec, _ in predictions[prot_id]]
        pred_ec_lists.append(pred_ecs)
        true_ec_lists.append(true_labels[prot_id])

    # Convert to binary matrix using MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=sorted(all_ecs))

    true_matrix = mlb.fit_transform(true_ec_lists)
    pred_matrix = mlb.transform(pred_ec_lists)

    # Compute metrics
    metrics = {
        'precision': precision_score(true_matrix, pred_matrix, average='weighted', zero_division=0),
        'recall': recall_score(true_matrix, pred_matrix, average='weighted', zero_division=0),
        'f1': f1_score(true_matrix, pred_matrix, average='weighted', zero_division=0),
        'accuracy': accuracy_score(true_matrix, pred_matrix),
        'n_samples': len(sorted_ids),
        'n_ecs': len(all_ecs),
    }

    return metrics


def print_eval_report(metrics: Dict[str, float]) -> None:
    """Print evaluation metrics report."""
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"  Samples:    {metrics['n_samples']}")
    print(f"  EC classes: {metrics['n_ecs']}")
    print("-" * 50)
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print("=" * 50)
