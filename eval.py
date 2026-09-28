"""Evaluate SFT / RLCD / temperature-scaled / base-rate models on held-out
in-distribution and OOD eval sets. Prints comparison tables and saves
reliability diagrams to plots/.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import data
import metrics
from model import CalibrationMLP

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


def load_model(name):
    model = CalibrationMLP()
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, name), map_location="cpu"))
    model.eval()
    return model


def predict(model, x):
    with torch.no_grad():
        return model(x).numpy()


def predict_temp_scaled(model, x, t):
    with torch.no_grad():
        z = model.logits(x)
        return torch.sigmoid(z / t).numpy()


def base_rate_predict(y_train_mean, n):
    return np.full(n, y_train_mean, dtype=np.float32)


def print_table(title, rows):
    cols = ["method", "brier", "log_loss", "ece", "reliability", "resolution",
            "uncertainty", "auroc", "accuracy"]
    widths = {c: max(len(c), 9) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])))

    print(f"\n--- {title} ---")
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = []
        for c in cols:
            v = r[c]
            line.append((f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(widths[c]))
        print(" | ".join(line))


def build_rows(preds_by_method, y):
    rows = []
    for name, p in preds_by_method.items():
        rep = metrics.full_report(p, y)
        rows.append({
            "method": name,
            "brier": rep["brier"],
            "log_loss": rep["log_loss"],
            "ece": rep["ece"],
            "reliability": rep["reliability"],
            "resolution": rep["resolution"],
            "uncertainty": rep["uncertainty"],
            "auroc": rep["auroc"],
            "accuracy": rep["accuracy"],
        })
    return rows, {name: metrics.full_report(p, y) for name, p in preds_by_method.items()}


def plot_reliability(reports, y_label, out_path):
    fig, (ax_rel, ax_cnt) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax_rel.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    bin_centers = None
    for name, rep in reports.items():
        edges = rep["bin_edges"]
        bin_centers = (edges[:-1] + edges[1:]) / 2
        mask = rep["bin_counts"] > 0
        ax_rel.plot(rep["bin_conf"][mask], rep["bin_acc"][mask], marker="o",
                    label=f"{name} (ECE={rep['ece']:.3f})")

    ax_rel.set_ylabel("Empirical accuracy")
    ax_rel.set_title(f"Reliability diagram — {y_label}")
    ax_rel.legend(fontsize=8)
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)

    width = (bin_centers[1] - bin_centers[0]) * 0.8 / len(reports)
    for i, (name, rep) in enumerate(reports.items()):
        offset = (i - len(reports) / 2) * width
        ax_cnt.bar(bin_centers + offset, rep["bin_counts"], width=width, label=name)
    ax_cnt.set_xlabel("Predicted confidence (bin)")
    ax_cnt.set_ylabel("Count")
    ax_cnt.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    splits = data.build_all_splits()
    x_train_np, y_train_np, _ = splits["train"]
    x_id, y_id, p_true_id = splits["eval_id"]
    x_ood, y_ood, p_true_ood = splits["eval_ood"]

    x_id_t, x_ood_t = torch.from_numpy(x_id), torch.from_numpy(x_ood)

    sft = load_model("sft.pt")
    rlcd = load_model("rlcd.pt")

    with open(os.path.join(CKPT_DIR, "training_log.json")) as f:
        log = json.load(f)
    temperature = log["temperature"]

    y_train_mean = float(y_train_np.mean())

    preds_id = {
        "base-rate": base_rate_predict(y_train_mean, len(y_id)),
        "SFT": predict(sft, x_id_t),
        "temp-scaled": predict_temp_scaled(sft, x_id_t, temperature),
        "RLCD": predict(rlcd, x_id_t),
    }
    preds_ood = {
        "base-rate": base_rate_predict(y_train_mean, len(y_ood)),
        "SFT": predict(sft, x_ood_t),
        "temp-scaled": predict_temp_scaled(sft, x_ood_t, temperature),
        "RLCD": predict(rlcd, x_ood_t),
    }

    rows_id, reports_id = build_rows(preds_id, y_id)
    rows_ood, reports_ood = build_rows(preds_ood, y_ood)

    print_table("In-distribution eval set", rows_id)
    print_table("OOD (covariate-shifted) eval set", rows_ood)

    print(f"\nECE bin scheme: {metrics.N_BINS} equal-width bins over [0, 1].")
    print(f"Temperature fitted on SFT validation logits: T={temperature:.4f}")

    print("\nGate decision from training:")
    for k, v in log["gate_report"].items():
        print(f"  {k}: {v}")

    print("\nBeta ablation (effect of beta on final KL / calibration, short runs):")
    for beta, r in log["beta_ablation"].items():
        print(f"  beta={beta:>4s}  final_mean_kl={r['final_mean_kl']:.5f}  "
              f"val_logloss={r['final_val_logloss']:.4f}  val_ece={r['final_val_ece']:.4f}")

    plot_reliability(reports_id, "in-distribution", os.path.join(PLOT_DIR, "reliability_id.png"))
    plot_reliability(reports_ood, "OOD (covariate shift)", os.path.join(PLOT_DIR, "reliability_ood.png"))

    # The Murphy decomposition BS = Reliability - Resolution + Uncertainty is
    # exact only when the forecast is constant within each bin. With a
    # continuous MLP output grouped into equal-width bins, within-bin
    # forecast dispersion leaves a small residual that shrinks as bin width
    # shrinks; it is not a bug. We check the residual stays small relative
    # to bin width rather than requiring exact equality.
    max_residual = 0.0
    for split_name, reports in [("ID", reports_id), ("OOD", reports_ood)]:
        for name, rep in reports.items():
            diff = abs(rep["brier"] - rep["reconstructed_bs"])
            max_residual = max(max_residual, diff)
            assert diff < 5e-3, f"Decomposition mismatch for {name}/{split_name}: {diff}"
    print(f"\nBrier decomposition identity (BS = Reliability - Resolution + Uncertainty) "
          f"holds within {max_residual:.5f} for all methods "
          f"(residual is within-bin forecast dispersion from {metrics.N_BINS} equal-width bins, not a bug).")


if __name__ == "__main__":
    main()
