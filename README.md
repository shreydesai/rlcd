# RLCD: Reinforcement Learning from Calibrated Decisions (toy experiment)

A small, fully synthetic PyTorch experiment testing whether an RLCD-style
bandit objective (proper-scoring-rule reward, REINFORCE gradient, KL penalty
to a reference policy) can improve probability calibration beyond a
supervised (SFT) baseline and a temperature-scaling baseline — without
trading away discrimination or breaking under covariate shift.

The model emits a calibrated probability in a single forward pass (no
autoregression, no sampling of network internals). The "action" in the
bandit formulation *is* the emitted probability itself.

## Why this setup

- Ground truth is known exactly (`p_true`), so calibration can be audited
  against the true data-generating process, not just against finite-sample
  outcomes.
- A degenerate constant base-rate predictor is included on purpose: it gets
  a great ECE while having zero resolution and 0.5 AUROC, which is why ECE
  alone is not a sufficient calibration metric — it must be read alongside
  discrimination (AUROC/accuracy) and the Brier decomposition (resolution
  in particular).
- An OOD (covariate-shifted) eval split is generated with the *same*
  labeling function but a shifted feature distribution, to check whether
  any calibration gains from RLCD survive distribution shift.

## Files

- `data.py` — synthetic data generator. `p_true = sigmoid(w·x + 0.5·x0·x1)`,
  `y ~ Bernoulli(p_true)`, `x ∈ R^10 ~ N(0, I)`. `w` is deterministically
  regenerated from a fixed seed (`WEIGHT_SEED`) so every run reconstructs
  the exact same ground truth. `build_all_splits()` produces:
  - `train` (20000, in-distribution)
  - `val` (4000, in-distribution — SFT model selection, temperature fit,
    ID half of the RLCD accept/reject gate)
  - `ood_val` (2000, covariate-shifted — OOD half of the gate)
  - `eval_id` / `eval_ood` (2500 + 2500, fresh draws, never touched during
    training — final reported numbers)
- `model.py` — `CalibrationMLP`: 10 → 64 → 64 → 1, ReLU, sigmoid head.
- `metrics.py` — Brier score, log loss, ECE (15 equal-width bins), Murphy
  Brier decomposition (Reliability − Resolution + Uncertainty), AUROC
  (rank-based, no sklearn dependency), accuracy.
- `train.py` — runs all three training phases and writes `checkpoints/`.
- `eval.py` — loads checkpoints, prints comparison tables, saves
  reliability diagrams to `plots/`.

## Training phases (`train.py`)

1. **SFT** — supervised warm start, minimizes `BCEWithLogitsLoss` directly.
   This is also `pi_ref` for the RLCD KL penalty.
2. **RLCD** — contextual bandit continuation of the SFT policy:
   - Forward pass gives a mean probability `mu_theta(x)`.
   - Sample the emitted probability `p_hat = mu_theta(x) + eps`,
     `eps ~ N(0, sigma^2)` (exploration jitter, `sigma = 0.05`), clamped to
     stay a valid probability.
   - Reward: `R(p, y) = -(p_hat - y)^2` (Brier reward).
   - REINFORCE gradient on `log N(p_hat; mu_theta(x), sigma)`, with an
     exponential-moving-average baseline (momentum 0.9) to reduce variance.
   - KL penalty: since `p_hat` is drawn from `N(mu_theta, sigma)` under the
     policy and `N(mu_ref, sigma)` under the frozen reference (same fixed
     `sigma`), `KL(pi_theta || pi_ref) = (mu_theta - mu_ref)^2 / (2*sigma^2)`
     in closed form.
   - Total objective: `L = -log_prob * (R - baseline) + beta * KL`,
     `beta = 0.1` by default.
   - At eval time the policy is used with **no jitter** — jitter only
     exists to generate the training signal; inference is a single
     deterministic forward pass, per the "one pass → probability" design.
3. **Beta ablation** — short (8-epoch) side-runs at
   `beta ∈ {0, 0.1, 0.5, 2.0}` starting from the same SFT weights, purely to
   log the effect of beta on final KL and validation ECE/log-loss. These
   checkpoints are discarded; only the numbers are logged
   (`checkpoints/training_log.json`, printed by `eval.py`). As expected,
   KL shrinks monotonically as beta grows (`beta=0` lets the policy drift
   far from `pi_ref`; `beta=2.0` clamps it almost in place).
4. **Accept/reject gate** — after the full RLCD run, the policy is compared
   against `pi_ref` on the in-distribution validation set and the OOD
   validation set (both held out from training, both distinct from the
   final `eval_id`/`eval_ood` sets so the gate decision can't leak into
   reported numbers). The update is **rejected** (reverting to SFT weights)
   if accuracy improved *and* either ECE or OOD Brier score got worse by
   more than a small tolerance (0.01) — i.e. if the RLCD step bought
   accuracy by damaging calibration or OOD behavior. The decision and the
   numbers behind it are always printed and logged, never silently
   accepted.
5. **Temperature scaling** — a single scalar `T` is fit by minimizing log
   loss (`BCEWithLogitsLoss(z/T, y)`, LBFGS) on the SFT model's validation
   logits. This is the cheap baseline RLCD has to beat.

## Metrics (`metrics.py`, computed by `eval.py`)

- `BS = (1/N) * sum_i (p_i - y_i)^2`
- `ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|`, 15 equal-width bins
  over `[0, 1]`.
- Murphy decomposition: `BS = Reliability - Resolution + Uncertainty`,
  where `Uncertainty = ybar*(1-ybar)`. This identity is **exact** only when
  the forecast is constant within each bin (discrete forecast categories);
  with a continuous MLP output grouped into equal-width bins there's a
  small residual from within-bin forecast dispersion. `eval.py` checks the
  residual stays small (< 5e-3) and reports it rather than asserting exact
  equality or silently ignoring it.
- AUROC via rank-sum (Mann-Whitney U), accuracy at threshold 0.5.
- A constant base-rate predictor (`p = mean(y_train)` for every input) is
  included as the degenerate baseline: near-zero ECE, near-zero resolution,
  0.5 AUROC.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python train.py   # writes checkpoints/{sft,rlcd,rlcd_unfiltered}.pt + training_log.json
python eval.py    # prints comparison tables, writes plots/reliability_{id,ood}.png
```

Everything is seeded (`SEED = 0` in `train.py`, fixed seeds per split in
`data.py`) — reruns are reproducible. No real data, no network calls.

## What the plots show

`plots/reliability_id.png` and `plots/reliability_ood.png` each have two
panels:
- **Top**: predicted confidence (x, bin mean) vs. empirical accuracy (y,
  bin mean) for base-rate / SFT / temp-scaled / RLCD, with the perfect-
  calibration diagonal. Curves that hug the diagonal are well calibrated;
  the legend reports each method's ECE.
- **Bottom**: bin occupancy counts. This is what makes the base-rate line
  degenerate obvious — all 2500 points land in a single bin (it always
  predicts the same probability), versus the other methods that spread
  predictions across the range and can be assessed bin-by-bin.

## What to look for in the comparison tables

`eval.py` prints two tables (in-distribution and OOD), each row one of
base-rate / SFT / temp-scaled / RLCD, columns Brier / log loss / ECE /
Reliability / Resolution / Uncertainty / AUROC / accuracy:

- Base-rate should have the best (lowest) ECE of all methods but ~0
  resolution and 0.5 AUROC — this is the point: **ECE alone can't tell you
  a model is useless**, you need resolution/AUROC alongside it.
- SFT vs. temp-scaled vs. RLCD should have comparable AUROC/accuracy
  (discrimination is set by the network's ranking ability, which none of
  these phases directly touch) — the interesting comparison is whether
  RLCD improves ECE/Reliability over SFT and temp-scaled, in-distribution
  *and* under the OOD shift, without regressing accuracy. The
  accept/reject gate report (also printed) states explicitly whether the
  RLCD run passed or failed that check on this run's seed.
