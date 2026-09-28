"""RLCD-style calibration training.

Three phases, run in order:
  1. SFT      - supervised warm start, minimize log loss directly.
  2. RLCD     - freeze the SFT model as pi_ref, continue training the policy
                as a contextual bandit: sample p_hat = mu_theta(x) + jitter,
                reward with the Brier score, REINFORCE gradient with a
                running baseline, plus a beta * KL(pi_theta || pi_ref)
                penalty. Ends with an accept/reject gate against pi_ref.
  3. TempScale - fit a single scalar T on the SFT model's validation logits.

All checkpoints and the training log are written to checkpoints/.
"""

import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

import data
import metrics
from model import CalibrationMLP

SEED = 0
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

SFT_EPOCHS = 30
SFT_LR = 1e-3
SFT_BATCH = 256

RLCD_EPOCHS = 30
RLCD_LR = 3e-4
RLCD_BATCH = 256
RLCD_BETA = 0.1
RLCD_JITTER_STD = 0.05
RLCD_BASELINE_MOMENTUM = 0.9
P_CLAMP_EPS = 1e-4

BETA_ABLATION = [0.0, 0.1, 0.5, 2.0]
BETA_ABLATION_EPOCHS = 8

GATE_ECE_TOL = 0.01
GATE_BRIER_TOL = 0.01


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_tensors(split):
    x, y, p_true = split
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(p_true)


def iterate_minibatches(n, batch_size, generator):
    order = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        yield order[i:i + batch_size]


def eval_logloss(model, x, y):
    model.eval()
    with torch.no_grad():
        p = model(x).clamp(1e-7, 1 - 1e-7)
        loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean().item()
    model.train()
    return loss


def train_sft(x_train, y_train, x_val, y_val):
    print("\n=== Phase 1: SFT (supervised log-loss warm start) ===")
    set_seed(SEED)
    model = CalibrationMLP()
    opt = torch.optim.Adam(model.parameters(), lr=SFT_LR)
    bce = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(SEED)

    history = []
    n = x_train.shape[0]
    for epoch in range(SFT_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for idx in iterate_minibatches(n, SFT_BATCH, gen):
            xb, yb = x_train[idx], y_train[idx]
            opt.zero_grad()
            loss = bce(model.logits(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        val_loss = eval_logloss(model, x_val, y_val)
        history.append({"epoch": epoch, "train_bce": epoch_loss / n_batches, "val_logloss": val_loss})
        if epoch % 5 == 0 or epoch == SFT_EPOCHS - 1:
            print(f"  epoch {epoch:3d}  train_bce={epoch_loss / n_batches:.4f}  val_logloss={val_loss:.4f}")

    return model, history


def kl_gaussian_means(mu_theta, mu_ref, sigma):
    """KL(N(mu_theta, sigma) || N(mu_ref, sigma)) for equal, fixed sigma."""
    return (mu_theta - mu_ref) ** 2 / (2 * sigma ** 2)


def run_rlcd_phase(sft_model, x_train, y_train, x_val, y_val,
                    beta, epochs, lr=RLCD_LR, batch_size=RLCD_BATCH,
                    jitter_std=RLCD_JITTER_STD, log_prefix=""):
    """Bandit RLCD training. Returns (policy_model, history)."""
    set_seed(SEED + 1)
    policy = copy.deepcopy(sft_model)
    ref = copy.deepcopy(sft_model)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(SEED + 1)

    baseline = None
    history = []
    n = x_train.shape[0]

    for epoch in range(epochs):
        policy.train()
        ep_reward, ep_kl, ep_loss, n_batches = 0.0, 0.0, 0.0, 0
        for idx in iterate_minibatches(n, batch_size, gen):
            xb, yb = x_train[idx], y_train[idx]

            mu_theta = policy(xb)
            with torch.no_grad():
                mu_ref = ref(xb)
                eps = torch.randn_like(mu_theta) * jitter_std
                p_hat = (mu_theta.detach() + eps).clamp(P_CLAMP_EPS, 1 - P_CLAMP_EPS)
                reward = -(p_hat - yb) ** 2  # Brier reward, R(p, y) = -(p - y)^2

            if baseline is None:
                baseline = reward.mean().item()
            advantage = reward - baseline

            log_prob = (-0.5 * ((p_hat - mu_theta) / jitter_std) ** 2
                        - math.log(jitter_std) - 0.5 * math.log(2 * math.pi))
            kl = kl_gaussian_means(mu_theta, mu_ref, jitter_std)

            pg_loss = -(log_prob * advantage).mean()
            kl_loss = kl.mean()
            loss = pg_loss + beta * kl_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            baseline = (RLCD_BASELINE_MOMENTUM * baseline
                        + (1 - RLCD_BASELINE_MOMENTUM) * reward.mean().item())

            ep_reward += reward.mean().item()
            ep_kl += kl_loss.item()
            ep_loss += loss.item()
            n_batches += 1

        val_loss = eval_logloss(policy, x_val, y_val)
        with torch.no_grad():
            val_p = policy(x_val).numpy()
        val_ece, *_ = metrics.ece(val_p, y_val.numpy())
        history.append({
            "epoch": epoch,
            "mean_reward": ep_reward / n_batches,
            "mean_kl": ep_kl / n_batches,
            "loss": ep_loss / n_batches,
            "val_logloss": val_loss,
            "val_ece": val_ece,
        })
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  {log_prefix}epoch {epoch:3d}  reward={ep_reward / n_batches:+.4f}  "
                  f"KL={ep_kl / n_batches:.5f}  val_logloss={val_loss:.4f}  val_ece={val_ece:.4f}")

    return policy, history


def beta_ablation(sft_model, x_train, y_train, x_val, y_val):
    print("\n=== Beta ablation (logging the effect of beta on KL / calibration) ===")
    results = {}
    for beta in BETA_ABLATION:
        print(f" -- beta={beta}")
        _, hist = run_rlcd_phase(sft_model, x_train, y_train, x_val, y_val,
                                  beta=beta, epochs=BETA_ABLATION_EPOCHS,
                                  log_prefix=f"[beta={beta}] ")
        final = hist[-1]
        results[str(beta)] = {
            "final_mean_kl": final["mean_kl"],
            "final_val_logloss": final["val_logloss"],
            "final_val_ece": final["val_ece"],
        }
    return results


def accept_reject_gate(sft_model, rlcd_model, x_val, y_val, x_ood_val, y_ood_val):
    print("\n=== RLCD accept/reject gate (vs. pi_ref = SFT) ===")
    with torch.no_grad():
        p_sft_val = sft_model(x_val).numpy()
        p_rlcd_val = rlcd_model(x_val).numpy()
        p_sft_ood = sft_model(x_ood_val).numpy()
        p_rlcd_ood = rlcd_model(x_ood_val).numpy()

    y_val_np, y_ood_np = y_val.numpy(), y_ood_val.numpy()

    acc_sft = metrics.accuracy(p_sft_val, y_val_np)
    acc_rlcd = metrics.accuracy(p_rlcd_val, y_val_np)
    ece_sft, *_ = metrics.ece(p_sft_val, y_val_np)
    ece_rlcd, *_ = metrics.ece(p_rlcd_val, y_val_np)
    brier_ood_sft = metrics.brier_score(p_sft_ood, y_ood_np)
    brier_ood_rlcd = metrics.brier_score(p_rlcd_ood, y_ood_np)

    accuracy_improved = acc_rlcd > acc_sft
    ece_worse = ece_rlcd > ece_sft + GATE_ECE_TOL
    ood_worse = brier_ood_rlcd > brier_ood_sft + GATE_BRIER_TOL
    reject = accuracy_improved and (ece_worse or ood_worse)

    report = {
        "acc_sft": acc_sft, "acc_rlcd": acc_rlcd,
        "ece_sft": ece_sft, "ece_rlcd": ece_rlcd,
        "brier_ood_sft": brier_ood_sft, "brier_ood_rlcd": brier_ood_rlcd,
        "accuracy_improved": accuracy_improved,
        "ece_worse": ece_worse,
        "ood_worse": ood_worse,
        "rejected": reject,
    }
    for k, v in report.items():
        print(f"  {k}: {v}")

    if reject:
        print("  >>> REJECTED: RLCD update bought accuracy by destroying calibration "
              "or OOD behavior. Reverting to SFT weights for the final policy.")
        final_model = copy.deepcopy(sft_model)
    else:
        print("  >>> ACCEPTED: RLCD update kept calibration and OOD behavior within tolerance.")
        final_model = rlcd_model

    return final_model, report


def fit_temperature(sft_model, x_val, y_val):
    print("\n=== Phase 3: Temperature scaling baseline ===")
    with torch.no_grad():
        z_val = sft_model.logits(x_val)

    log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t), start at T=1
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t)
        loss = bce(z_val / t, y_val)
        loss.backward()
        return loss

    opt.step(closure)
    t_final = torch.exp(log_t).item()
    val_loss = bce(z_val / t_final, y_val).item()
    print(f"  fitted T={t_final:.4f}  val_logloss_after_scaling={val_loss:.4f}")
    return t_final


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    set_seed(SEED)

    print("Generating synthetic data...")
    splits = data.build_all_splits()
    x_train, y_train, _ = to_tensors(splits["train"])
    x_val, y_val, _ = to_tensors(splits["val"])
    x_ood_val, y_ood_val, _ = to_tensors(splits["ood_val"])

    sft_model, sft_history = train_sft(x_train, y_train, x_val, y_val)
    torch.save(sft_model.state_dict(), os.path.join(CKPT_DIR, "sft.pt"))

    ablation_results = beta_ablation(sft_model, x_train, y_train, x_val, y_val)

    print(f"\n=== Phase 2: RLCD (beta={RLCD_BETA}) ===")
    rlcd_model, rlcd_history = run_rlcd_phase(
        sft_model, x_train, y_train, x_val, y_val,
        beta=RLCD_BETA, epochs=RLCD_EPOCHS,
    )

    final_model, gate_report = accept_reject_gate(
        sft_model, rlcd_model, x_val, y_val, x_ood_val, y_ood_val,
    )
    torch.save(final_model.state_dict(), os.path.join(CKPT_DIR, "rlcd.pt"))
    torch.save(rlcd_model.state_dict(), os.path.join(CKPT_DIR, "rlcd_unfiltered.pt"))

    temperature = fit_temperature(sft_model, x_val, y_val)

    log = {
        "seed": SEED,
        "sft_history": sft_history,
        "beta_ablation": ablation_results,
        "rlcd_beta": RLCD_BETA,
        "rlcd_history": rlcd_history,
        "gate_report": gate_report,
        "temperature": temperature,
    }
    with open(os.path.join(CKPT_DIR, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nSaved checkpoints + training_log.json to {CKPT_DIR}/")


if __name__ == "__main__":
    main()
