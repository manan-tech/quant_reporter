# quant_reporter — Development TODO (council-vetted, 2026-06-02)

Finalized after the LLM council pressure-test. Full reasoning:
`council-report-20260602-181128.html` / `council-transcript-20260602-181128.md`.

**Core problem:** `recommend_weights()` (`recommendation.py:50`) feeds **raw historical
mean + sample covariance** into max-Sharpe → in-sample-fragile advice, despite robust
estimators (Ledoit-Wolf, denoising, Black-Litterman, HRP, risk-parity) already shipping
unused. The **mean** is the dominant source of optimizer garbage, not just the covariance.

Each item gets its own brainstorm → spec → plan → execute cycle (`docs/superpowers/`).

---

## 0. Lock a baseline  ← DO THIS FIRST
- [ ] Run the existing walk-forward harness (`walk_forward_recommendation` /
      `_rolling_oos_sharpe`) on today's raw-mean/sample-cov path.
- [ ] Record baseline **OOS Sharpe + degradation** on a **held-out split we commit to
      NEVER tuning against** (guards against overfitting our own validation layer).
- [ ] This number is the scoreboard every later change must beat.
- *Cost: ~1 afternoon, zero new code.*

## 1. Estimation upgrade  (honest estimates + visible uncertainty)
### 1a. Covariance
- [ ] Wire **Ledoit-Wolf** (`ledoit_wolf_covariance`) + **denoising**
      (`get_optimization_inputs(denoise_cov=True)`) into `recommend_weights`.
- [ ] Re-run harness, compare vs baseline. (Cheap, proven, biggest single OOS win.)
### 1b. Expected returns  (the real fix)
- [ ] Add a **no-forecast default path** using already-shipped `min_variance` /
      `risk_parity` / `max_diversification` — the honest "we don't forecast returns" stance.
- [ ] Add **Black-Litterman equilibrium returns** (`calculate_black_litterman_posterior`)
      as the expected-return prior. NOTE: BL is *not* a one-line swap (needs prior +
      tau/omega; with no views it collapses to the benchmark — that's a product decision).
- [ ] Defer full "views as a product" API to a later expansion.
### 1c. Uncertainty as a first-class output
- [ ] Attach the **OOS-degradation number** to every recommendation (user-facing).
- [ ] Add resampled / CI **weight bands** ("how much to trust this weight").

## 2. Hard RiskLimits  (ship ALONGSIDE item 1, not after)
- [ ] Configurable `RiskLimits`: concentration / vol / CVaR caps as a **hard backstop**
      that protects regardless of estimate quality.
- [ ] Surface downside *with* the weights ("how badly can this lose?") — trust = legibility.
- [ ] Scenario & stress testing + structured alert engine can follow.

## 3. Transaction-cost-aware optimization  (NEW pillar — prerequisite for tax)
- [ ] Add a transaction-cost term to the optimization objective. Cheap, and it's what
      makes rebalancing (and later tax-aware) advice credible.

## 4. Tax-aware rebalancing  (LAST — lowest leverage)
- [ ] The "T" in TTLU. Build on top of transaction-cost-aware optimization.

---

## CUT (false precision / overfit / reputational liability)
- ❌ Kalman dynamic beta
- ❌ Gaussian Process return/vol forecasts
- ❌ ML / random-forest return forecasts

## PARKED (maybe, only if narratable + exposed as a stated assumption)
- ⏸️ Regime-switching (Markov) moments

---

## Cross-cutting guardrails
- [ ] **Never tune estimators on the split we report** — separate selection from validation.
- [ ] Every recommendation ships with plain-language "why / assumption / what breaks it".
- [ ] Keep a loud **"this is not investment advice"** honesty layer (liability).
- [ ] Question whether max-Sharpe should even be the default engine — the shipped
      min-variance / risk-parity / HRP sidestep mean estimation entirely.

## Separately gated (from HANDOFF.md, not dev work)
- [ ] Publish 2.2.0 to PyPI (~week of 2026-06-08): bump `__version__`, move
      `[Unreleased]`→`[2.2.0]`, wheel-smoke-test, `twine upload`.
- [ ] Revoke the old (compromised) PyPI token; regenerate project-scoped.
