# Why our accuracy can be better than full-data training

**Summary for the paper:** We can beat “always retrain on all data” when the stream has **recurring regimes** and we evaluate on the **current regime**. The full-data model is optimal for the **mixture** of all past regimes; we use a model optimized for the **current regime’s distribution** (a specialist). On data from that regime, the specialist can have lower error than the generalist.

---

## The puzzle

How can a model that reuses a checkpoint (trained on a subset of data) beat a model retrained on **all** accumulated data? Isn’t more data always better?

## The answer

**Full-data training fits one model to a *mixture* of regimes. We use a *specialist* for the current regime. On data from that regime, the specialist can have lower loss than the mixture-optimal model.**

---

## What full retrain actually optimizes

- At each step, the baseline fits a single model on **all data so far**: regime 1 + regime 2 + … + current batch.
- So it minimizes loss over the **mixture distribution**
  - \( P_{\text{mix}} = \frac{n_1}{N} P_1 + \frac{n_2}{N} P_2 + \cdots \)
  - where \( P_k \) is the distribution of regime \( k \) and \( n_k \) is the number of points from regime \( k \).
- The resulting model is **optimal for predicting under \( P_{\text{mix}} \)**, not for predicting under the **current** regime’s distribution \( P_{\text{current}} \).

## Why the mixture-optimal model is worse on the current regime

- When the **test batch** is from **one** regime (say A), we care about error under \( P_A \).
- The **Bayes-optimal predictor for \( P_A \)** (e.g. for MSE, the conditional mean \( E[Y \mid \text{regime A}] \)) minimizes error on that regime.
- The **full-data model** is tuned to minimize loss over the mixture, so it behaves like a **compromise** over all regimes. That compromise is in general **not** the best predictor for regime A alone.
- So on data from A:
  - **Ours:** We load the checkpoint trained when we last saw regime A → a model optimized for \( P_A \) (specialist).
  - **Full retrain:** Uses the model optimized for \( P_{\text{mix}} \) (generalist). On data from A, its error can be **higher** than the specialist’s.

## Minimal math (MSE, two regimes)

- Regimes A and B with means \( \mu_A \), \( \mu_B \), same variance \( \sigma^2 \). Test data from A.
- **Specialist (ours):** predict \( \mu_A \) → MSE = \( \sigma^2 \).
- **Full retrain:** optimal for mixture is \( \mu_{\text{mix}} = \frac{n_A \mu_A + n_B \mu_B}{N} \). On test data \( Y \sim P_A \):
  - MSE = \( E[(Y - \mu_{\text{mix}})^2] = \sigma^2 + (\mu_A - \mu_{\text{mix}})^2 \geq \sigma^2 \),
  - with equality only if \( \mu_{\text{mix}} = \mu_A \). So the full-data predictor has **strictly higher** MSE on the current regime whenever the regime mean differs from the mixture mean.

---

## How this matches our experiments

- **Multi_regime synthetic data:** 4 regimes (different base, trend, seasonal amplitude, noise). Regimes recur over time.
- **Full retrain:** One GRU fitted on all points so far (mix of regimes 1–3, etc.) → compromise model.
- **Ours:** When the new batch’s distribution matches a stored regime, we load the checkpoint trained on that regime → specialist for the current regime.
- **Result:** On all 7 multi_regime datasets, regime-aware reuse **wins vs both** full retrain and TTA: we deploy the right specialist; full retrain keeps using the compromise; TTA adapts only on the small new batch and can underfit or drift.

---

## One-sentence takeaway for the paper

*We can beat full-data training when the stream has recurring regimes and we evaluate on the current regime: the full-data model is optimal for the mixture of all past regimes, while we use a model optimized for the current regime’s distribution.*

---

See also: `notebooks/why_regime_wins_analysis.ipynb` (Section 4 and numerical cell) for the same reasoning and a small MSE comparison.
