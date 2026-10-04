# ReturnIQ engine: measured proof

**Every figure below is simulated, on planted synthetic patterns.** Nothing here was measured on a
real seller. The generator plants wardrobing-style fast returners, risky pincodes, high-return SKUs
and low-intent COD buyers, so the engine is being tested on structure we put there. Treat the
numbers as a check that the engine finds planted structure, not as evidence of real-world lift.

How to reproduce (venv at `.venv`):

```
python -m returniq_engine.report --seed 42 --eval --perf
python -m returniq_engine.report --seed 42 --n 10000 --no-model
pytest packages/engine -q -s
```

Dataset: seed 42, 1,200 orders unless stated (dataset_version `02d3194806f96730`), rules
`rules-1.0.0`, generator spec `gen-1.0.0`. Nothing was tuned to pass a test: the generator has not
changed since the P0 tests passed, and the rules points table did not need adjusting to keep HIGH
below 20% of orders.

## 1. Headline analytics (simulated, seed 42, n = 1,200)

| Metric | Value |
|---|---|
| RTO rate | 129 / 1,200 = 10.75% (pending 0) |
| Post-delivery return rate (accepted) | 246 / 1,047 delivered = 23.5% (requested 261 = 24.93%) |
| Flagged returns (rules, Stage 2 >= MEDIUM, reason-eligible) | 67 |
| Estimated PRR (estimate) | 6.4% of delivered orders |
| Observed operating loss, order value excluded | Rs 54,475.80 (all cost fields assumed defaults) |
| Est. net avoidable loss low / base / high (scenario) | Rs 814 / 3,961 / 7,108 |
| Median TTR | 16.56 h; LT_6H 91, H6_24 57, D1_3 58, D3_7 33, GT_7D 16, UNKNOWN 6 |
| Readiness | ACTIVE |

RTO and returns use different denominators (resolved dispatched shipments vs delivered orders) and
are never combined.

## 2. Band tables

Observed rate = share of orders in the band with the outcome. Lift = band rate / overall rate.

### Stage 1 (PRE_DISPATCH), outcome = RTO, rules only, seed 42, n = 1,200

| Band | Orders | % orders | Observed RTO rate | Lift |
|---|---|---|---|---|
| LOW | 1,032 | 86.0% | 9.2% | 0.86x |
| MEDIUM | 125 | 10.4% | 20.8% | 1.93x |
| HIGH | 43 | 3.6% | 18.6% | 1.73x |
| INSUFFICIENT_DATA | 0 | 0.0% | n/a | n/a |

At n = 1,200 HIGH (18.6%) is slightly below MEDIUM (20.8%): only 43 orders, so this is noise, but
it is not monotone and we report it as measured. At n = 10,000 (seed 42) the bands are monotone:

| Band | Orders | % orders | Observed RTO rate | Lift |
|---|---|---|---|---|
| LOW | 8,685 | 86.9% | 9.7% | 0.83x |
| MEDIUM | 902 | 9.0% | 23.2% | 2.00x |
| HIGH | 413 | 4.1% | 26.9% | 2.32x |

Stage 1 rules AUC on the RTO label is modest: 0.61 (n = 1,200, all rows) and 0.64 (n = 10,000, all
rows). HIGH precision is limited because the RTO signal available before dispatch is mostly history
(prior RTOs) and pincode lift, and a customer's first RTO cannot be anticipated from history.

### Stage 2 (POST_DELIVERY), outcome = planted preventable return, rules only, seed 42, n = 1,200 (261 return requests)

| Band | Returns | % | Observed preventable-return rate | Lift |
|---|---|---|---|---|
| LOW | 191 | 73.2% | 60.2% | 0.88x |
| MEDIUM | 35 | 13.4% | 91.4% | 1.33x |
| HIGH | 29 | 11.1% | 96.6% | 1.41x |
| INSUFFICIENT_DATA | 6 | 2.3% | 66.7% | 0.97x |

Read this carefully: the base rate of planted preventable returns among all return requests is high
(about 69%), so lift is capped near 1.45x. The LOW band still contains 60% planted-preventable
returns: rules miss planted returns that come from first-time customers in risky pincodes or on
high-return SKUs, because those have no customer history. Rules are a conservative precision tool
here, not a recall tool.

### Rules + ML blend (0.5 / 0.5), held-out test slice (last 20% by time), seed 42, n = 1,200

Stage 2 (53 test rows):

| Band | Returns | % | Observed preventable-return rate |
|---|---|---|---|
| LOW | 26 | 49.1% | 42.3% |
| MEDIUM | 7 | 13.2% | 85.7% |
| HIGH | 20 | 37.7% | 100.0% |

Stage 1 (240 test rows): LOW 226 (93.8%, 7.5% RTO), MEDIUM 15 (6.2%, 20.0%), HIGH 0. The 0.5/0.5 blend
never reached 70 on this slice, so Stage 1 HIGH is empty. Test slices this small
have wide error bars; use the 10,000-order evaluation below for model comparisons.

## 3. Model vs rules (time-based split: train earliest 70%, validate next 10%, test last 20%)

Model: LightGBM, isotonic calibration on the validation slice. Labels: the generator's hidden
labels (Stage 1 = RTO, Stage 2 = planted preventable return), used for training and evaluation on
synthetic data only; they never reach CSV or features. So the ML advantage below partly reflects
learning the planted structure (for example the lift features and `reason_eligible`), and says
nothing about real sellers.

| Seed | n | Stage | ML AUC | Rules AUC | P@12% | R@12% |
|---|---|---|---|---|---|---|
| 42 | 1,200 | Stage 1 | 0.7373 | 0.6090 | 0.172 | 0.250 |
| 42 | 1,200 | Stage 2 | 0.8547 | 0.7939 | 1.000 | 0.189 |
| 41 | 10,000 | Stage 1 | 0.7779 | 0.6662 | 0.340 | 0.331 |
| 41 | 10,000 | Stage 2 | 0.9203 | 0.8603 | 1.000 | 0.186 |
| 42 | 10,000 | Stage 1 | 0.7810 | 0.6802 | 0.349 | 0.377 |
| 42 | 10,000 | Stage 2 | 0.9443 | 0.8383 | 1.000 | 0.182 |
| 43 | 10,000 | Stage 1 | 0.8024 | 0.6857 | 0.402 | 0.393 |
| 43 | 10,000 | Stage 2 | 0.9101 | 0.8359 | 0.960 | 0.167 |

- Non-inferiority (ML AUC >= rules AUC - 0.02) holds in all 8 rows. ML beats rules by 0.06 to 0.12
  AUC here. Not tuned: the model and generator were not adjusted after seeing these numbers.
- "% preventable captured at top 12%" equals R@12 for the Stage 2 label: 18.9% (n = 1,200) and
  16.7% to 18.6% (n = 10,000). It is bounded above by about 12% of returns divided by the
  preventable share (about 69%), roughly 0.17 to 0.19, so the model is near the ceiling at that
  cut. P@12 = 1.0 at Stage 2 is similarly a consequence of the high base rate.
- Stage 2 top-25% policy simulation (rules vs ML) captured the same 28.7% of observed
  reason-eligible returns at n = 1,200, so the ML gain shows in AUC, not in that cut.

## 4. INSUFFICIENT_DATA share and HIGH share (rules only, seed 42, n = 1,200)

| Stage | INSUFFICIENT_DATA | HIGH |
|---|---|---|
| Stage 1 | 0 / 1,200 = 0.0% | 43 / 1,200 = 3.6% |
| Stage 2 | 6 / 261 = 2.3% | 29 / 261 = 11.1% |

Targets: INSUFFICIENT_DATA below 15% (met), HIGH at most about 20% (met). All 6 Stage 2
INSUFFICIENT_DATA cases are returns with no `delivered_at` (2% missing by design, plus scenario 6).

## 5. Scenarios (seed 42)

| # | Scenario | Result |
|---|---|---|
| 1 | Normal prepaid order | Stage 1 LOW (0), NORMAL_FULFILMENT |
| 2 | COD, 2 earlier RTOs | Stage 1 HIGH (92), OTP_VERIFICATION, alternatives PREPAID_ONLY / HOLD_FULFILMENT |
| 3 | Defective return in 3 h | Stage 2 LOW (0), fault-reason signal, INSTANT_RETURN |
| 4 | Repeat fast returner | Stage 2 HIGH (87), MANUAL_REVIEW, alternative REFUND_AFTER_QC |
| 5 | Late return, first-time customer | Stage 2 LOW (15), cold-start signals |
| 6 | Missing delivery timestamp | Stage 2 INSUFFICIENT_DATA, TTR unknown |
| 7 | Scenario-4 customer orders again | Stage 1 HIGH (72) because of the earlier flag |

Scenario 4's recommended action is MANUAL_REVIEW with REFUND_AFTER_QC as the first alternative
(the 0.6 example ordering). The test asserts `REFUND_AFTER_QC` is offered.

## 6. Performance (measured on the development machine, Apple silicon, Python 3.12)

| Measurement | Result |
|---|---|
| Generate 25,000 orders | 0.42 s |
| Build canonical frames (25,000) | 0.84 s to 1.4 s |
| `Dataset.prepare()` (25,000, all point-in-time features, both stages) | 0.14 s |
| `assess_batch` on 25,000 orders (Stage 1, objects built) | 1.09 s to 1.24 s across logged runs (limit 5 s) |
| `assess()` over 1,000 calls | p50 0.044 to 0.045 ms, p95 0.061 to 0.069 ms |
| Train one LightGBM stage model, n = 10,000 (3 runs each, `verification.md`) | Stage 1: 0.08 to 0.14 s; Stage 2: 0.06 s |

`assess_batch` excludes `prepare()`, which runs once per dataset version. A Render instance will be
slower than this machine; the 5 s limit leaves roughly 4x headroom. These timings exclude network.

## 7. Known limits

- All of the above is synthetic. Real-seller performance is unknown until Vedansh's seller data or
  outcomes arrive.
- Costs are assumed defaults; the net-avoidable figures are scenarios, not savings.
- Stage 1 is weak on RTO at small n; its value is mainly the prior-RTO and pincode signals.
- The ML component is trained on planted labels. With real data, `train()` needs seller outcomes.
- Rules-based flags used in Analytics and in the "next one" loop use the default MEDIUM threshold of
  40 and ignore the ML model, so they do not change when a model is retrained.
