"""Calibration / discrimination metrics shared by train.py's accept/reject
gate and eval.py's final report. Pure numpy, no sklearn dependency.
"""

import numpy as np

N_BINS = 15  # equal-width bins over [0, 1], shared everywhere ECE is used


def brier_score(p, y):
    """BS = (1/N) sum_i (p_i - y_i)^2"""
    p, y = np.asarray(p), np.asarray(y)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps=1e-7):
    """-(1/N) sum_i y_i*log(p_i) + (1-y_i)*log(1-p_i)"""
    p = np.clip(np.asarray(p), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _bin_stats(p, y, n_bins=N_BINS):
    """Equal-width bins over [0, 1]. Returns per-bin count, mean predicted
    confidence, and mean empirical accuracy/outcome rate."""
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)

    counts = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    acc = np.zeros(n_bins)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            conf[b] = p[mask].mean()
            acc[b] = y[mask].mean()
    return edges, counts, conf, acc


def ece(p, y, n_bins=N_BINS):
    """ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|, equal-width bins."""
    edges, counts, conf, acc = _bin_stats(p, y, n_bins)
    n = counts.sum()
    weights = counts / n
    return float(np.sum(weights * np.abs(acc - conf))), edges, counts, conf, acc


def brier_decomposition(p, y, n_bins=N_BINS):
    """Murphy (1973) decomposition: BS = Reliability - Resolution + Uncertainty.

    Reliability = sum_b (n_b/N) (conf_b - acc_b)^2
    Resolution  = sum_b (n_b/N) (acc_b - ybar)^2
    Uncertainty = ybar * (1 - ybar)
    """
    edges, counts, conf, acc = _bin_stats(p, y, n_bins)
    n = counts.sum()
    weights = counts / n
    ybar = np.asarray(y).mean()

    reliability = float(np.sum(weights * (conf - acc) ** 2))
    resolution = float(np.sum(weights * (acc - ybar) ** 2))
    uncertainty = float(ybar * (1 - ybar))
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "reconstructed_bs": reliability - resolution + uncertainty,
    }


def auroc(p, y):
    """Rank-based AUROC (Mann-Whitney U), no sklearn dependency."""
    p, y = np.asarray(p), np.asarray(y)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p)
    sorted_p = p[order]
    ranks_sorted = np.empty(len(p))
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks_sorted[i:j + 1] = avg_rank
        i = j + 1
    ranks = np.empty(len(p))
    ranks[order] = ranks_sorted

    sum_ranks_pos = ranks[y == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def accuracy(p, y, threshold=0.5):
    p, y = np.asarray(p), np.asarray(y)
    return float(np.mean((p >= threshold).astype(np.float32) == y))


def full_report(p, y, n_bins=N_BINS):
    ece_val, edges, counts, conf, acc = ece(p, y, n_bins)
    decomp = brier_decomposition(p, y, n_bins)
    return {
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
        "ece": ece_val,
        "auroc": auroc(p, y),
        "accuracy": accuracy(p, y),
        **decomp,
        "bin_edges": edges,
        "bin_counts": counts,
        "bin_conf": conf,
        "bin_acc": acc,
    }
