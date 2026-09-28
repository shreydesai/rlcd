"""Synthetic binary-decision data generator with a known ground-truth p_true.

p_true(x) = sigmoid(w . x + 0.5 * x0 * x1)
y ~ Bernoulli(p_true(x))

The weight vector w is derived deterministically from WEIGHT_SEED so every
run (and every audit of calibration-against-truth) reconstructs the exact
same ground truth.
"""

import numpy as np

N_FEATURES = 10
WEIGHT_SEED = 12345

# Feature-shift applied to the OOD half of the eval set: shifts the mean of
# every feature and rescales its spread, so the covariate distribution
# genuinely differs from training while keeping the same functional form
# for p_true (only the input distribution shifts, not the labeling rule).
OOD_MEAN_SHIFT = 1.25
OOD_SCALE = 1.6


def true_weights():
    """Deterministically regenerate the ground-truth weight vector w."""
    rng = np.random.default_rng(WEIGHT_SEED)
    w = rng.normal(loc=0.0, scale=0.8, size=N_FEATURES)
    return w


def p_true(x, w):
    """x: (N, 10), w: (10,) -> p_true: (N,)"""
    linear = x @ w
    nonlinear = 0.5 * x[:, 0] * x[:, 1]
    z = linear + nonlinear
    return 1.0 / (1.0 + np.exp(-z))


def generate_dataset(n, seed, ood=False):
    """Draw n fresh (x, y, p_true) triples.

    ood=True shifts the mean/scale of the feature distribution (covariate
    shift) while keeping the exact same labeling function p_true, so any
    calibration break under OOD reflects the model, not a change in the
    ground-truth rule.
    """
    rng = np.random.default_rng(seed)
    w = true_weights()
    x = rng.normal(size=(n, N_FEATURES))
    if ood:
        x = x * OOD_SCALE + OOD_MEAN_SHIFT
    p = p_true(x, w)
    y = rng.binomial(1, p).astype(np.float32)
    return x.astype(np.float32), y, p.astype(np.float32)


def build_all_splits(train_n=20000, eval_n=5000, val_n=4000, ood_val_n=2000,
                      train_seed=0, val_seed=1, ood_val_seed=2, eval_seed=3):
    """Build every split used across train.py / eval.py.

    - train:      train_n samples, in-distribution, for SFT + RLCD training.
    - val:        val_n samples, in-distribution, held out from training.
                  Used for SFT model selection, temperature scaling, and as
                  the in-distribution half of the RLCD accept/reject gate.
    - ood_val:    ood_val_n samples, covariate-shifted. Used only as the
                  OOD half of the RLCD accept/reject gate (never reported
                  in final eval numbers).
    - eval_id:    eval_n // 2 samples, in-distribution, fresh draw. Final
                  reported metrics.
    - eval_ood:   eval_n // 2 samples, covariate-shifted, fresh draw. Final
                  reported OOD metrics.
    """
    x_train, y_train, p_train = generate_dataset(train_n, train_seed, ood=False)
    x_val, y_val, p_val = generate_dataset(val_n, val_seed, ood=False)
    x_ood_val, y_ood_val, p_ood_val = generate_dataset(ood_val_n, ood_val_seed, ood=True)

    half = eval_n // 2
    x_eval_id, y_eval_id, p_eval_id = generate_dataset(half, eval_seed, ood=False)
    x_eval_ood, y_eval_ood, p_eval_ood = generate_dataset(half, eval_seed + 1000, ood=True)

    return {
        "train": (x_train, y_train, p_train),
        "val": (x_val, y_val, p_val),
        "ood_val": (x_ood_val, y_ood_val, p_ood_val),
        "eval_id": (x_eval_id, y_eval_id, p_eval_id),
        "eval_ood": (x_eval_ood, y_eval_ood, p_eval_ood),
    }


if __name__ == "__main__":
    splits = build_all_splits()
    for name, (x, y, p) in splits.items():
        print(f"{name:10s} x={x.shape} y_mean={y.mean():.3f} p_true_mean={p.mean():.3f}")
    print("true w:", true_weights())
