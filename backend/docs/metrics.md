# ReturnIQ metrics and scoring reference

Engine: `returniq_engine`, rules `rules-1.0.0`. Everything here is implemented in `packages/engine`
and checked by `pytest packages/engine`. Money is integer paise in the engine and API.
`risk_score` is a **risk index** from 0 to 100. It is not a probability.

All figures in the worked examples come from the synthetic seed-42 dataset (1,200 orders) and are
**simulated, on planted synthetic patterns**.

## 1. Metric definitions

| Metric | Formula | Notes |
|---|---|---|
| RTO rate | RTO shipments / resolved dispatched shipments x 100 | Resolved = DELIVERED, RTO, NDR, LOST. In-transit shipments are reported separately as `pending`. |
| Post-delivery return rate | Delivered orders with an accepted return / delivered orders x 100 | Requested returns are reported separately (`requested`, `requested_rate`). |
| Flagged return | Stage 2 assessment at `as_of = return_requested_at` is at least MEDIUM, and the reason is eligible | Analytics uses the rules component only. |
| Reason-eligible | Reason is not `DEFECTIVE`, `DAMAGED_IN_TRANSIT` or `WRONG_ITEM` | Seller and carrier faults are never preventable. |
| Estimated PRR | Flagged and eligible returns / delivered orders x 100 | Always an estimate: the share the system would have flagged. |
| Confirmed PRR | Seller-confirmed preventable returns / delivered orders x 100 | `null` until confirmations exist. Unknown stays unknown. |
| Observed operating loss | forward ship + reverse ship + packaging + handling + write-off - cost recovery | Accepted returns only. **Order value is excluded**: a refund is not automatically a loss. RTO loss is reported separately as `rto_observed_paise`. |
| Est. net avoidable loss | eligible loss x effectiveness - intervention cost - friction margin loss | Scenario (low / base / high = 20 / 40 / 60 %). Can be negative. Not savings. |
| Loss avoided per 100 orders | net avoidable / orders x 100 | Secondary metric. |
| Time to return (TTR) | `return_requested_at - delivered_at` in hours | Missing `delivered_at` gives `null` (never 0). Negative values are quarantined. |
| TTR buckets | `LT_6H` (<6), `H6_24` (6 to <24), `D1_3` (24 to <72), `D3_7` (72 to 168), `GT_7D` (>168), `UNKNOWN` | A fast return is evidence, not proof. |

RTO and post-delivery returns are never summed or charted on one axis. `Analytics` has no
combined field.

Assumed defaults (Settings, labelled "assumed"): forward ship Rs 60, reverse ship Rs 80,
packaging Rs 15, handling Rs 25, write-off 10% of order value when QC is `USED` or
`DAMAGED_BY_CUSTOMER`, intervention cost Rs 10 per flagged return, friction margin loss 2% of the
value of flagged orders, return window 7 days, minimum confidence 0.3. Cost fields that the CSV
does not supply are listed in `loss.assumed_fields`. Cost recovery that is not supplied is
treated as 0 and listed as assumed.

## 2. Feature availability by stage (leakage rule)

| Feature group | Stage 1 `as_of = order_date` | Stage 2 `as_of = return_requested_at` |
|---|---|---|
| Current order: payment mode, value, SKU, category, pincode | yes | yes |
| Customer history (orders, RTOs, returns, TTRs) strictly before `as_of` | yes | yes |
| SKU / category / pincode lift from events strictly before `as_of` | yes | yes |
| Current shipment: `delivered_at`, attempts | no | yes |
| Current return: TTR, `return_reason` | no | yes |
| `qc_result`, `return_accepted_at`, refund and cost fields | no | no |
| Any event after `as_of`, hidden labels | no | no |

How it is enforced:

- `STAGE1_FEATURES`, `STAGE2_FEATURES` and `FORBIDDEN_FEATURES` in `features.py` are explicit lists;
  the feature frames contain exactly those columns.
- History is computed from sorted event tables with `searchsorted(side="left")` plus cumulative sums,
  so an event at exactly `as_of` is excluded. The order's own events are subtracted.
- An RTO or failed delivery becomes known 7 days after dispatch (the CSV has no RTO timestamp).
- Tests: `test_leakage.py` mutates every event after `as_of` (other orders, later returns, deliveries
  and the order's own QC, refund and cost fields) and asserts the assessment is unchanged;
  `test_features.py` compares the vectorised features with a slow brute-force reference.
- Return-frequency history counts reason-eligible returns only, so a customer with several genuine
  defect returns is not penalised.

## 3. Rule points table (`rules-1.0.0`)

The rule component is `min(sum of points, 100)`.

| Code | Stage | Points |
|---|---|---|
| `COD_WITH_PRIOR_RTO` | 1, 2 | 30 with 1 earlier RTO, 40 with 2 or more (COD or partial COD) |
| `RTO_RATE_HIGH` | 1 | 20 when at least 2 earlier orders and at least half ended in RTO |
| `FAILED_DELIVERY_HISTORY` | 1 | 10 for 2 or more failed deliveries, 5 for 1 |
| `RETURN_FREQ_HIGH` | 1, 2 | 25 when the return rate is at least 0.5 (12 when at least 0.3), with at least 3 earlier orders |
| `RETURN_FREQ_RECENT` | 1, 2 | 10 for 2 or more eligible returns in the last 30 days |
| `FAST_RETURN_HISTORY` | 1, 2 | 15 when at least 2 eligible returns and at least half were under 6 h |
| `FLAGGED_PRIOR_RETURN` | 1 | 15 when an earlier return was flagged (the "next one" loop) |
| `RISKY_PINCODE_LIFT` | 1, 2 | up to 15, linear from lift 1.2 to 2.0 (Stage 1 uses RTO lift, Stage 2 return lift) |
| `HIGH_RETURN_SKU` | 1, 2 | up to 10, linear from lift 1.3 to 2.2 |
| `HIGH_RETURN_CATEGORY` | 1, 2 | up to 6, linear from lift 1.15 to 1.5 |
| `COD_HIGH_VALUE_VS_CATEGORY` | 1, 2 | 8 when COD and value is at least 1.5x the category median |
| `COD_NEW_CUSTOMER` | 1 | 5 for a first COD order |
| `TTR_ABNORMALLY_FAST` | 2 | 20 when TTR is under 6 h and under 0.5x the seller median |
| `SAME_CATEGORY_REPEAT` | 2 | 10 for 2 or more earlier eligible returns in the category, 5 for 1 |
| `POLICY_WINDOW_VIOLATION` | 2 | 15 when TTR exceeds the return window |
| `SELLER_OR_CARRIER_FAULT_REASON` | 2 | 0 points; zeroes every behavioural point for that return |

Lift smoothing: `lift = (k + count) / (k + expected)` with `k = 5` for pincode and SKU and
`k = 10` for category, where `expected = orders_for_key x seller baseline rate`. Small pincodes stay
near 1.0. With fewer than 50 baseline orders the lift is neutral (1.0). A missing pincode gives a
missing lift, not 1.0, and is listed in `missing_fields`.

The seller median TTR is an expanding median of earlier returns; with fewer than 10 it falls back to
48 h (assumed).

## 4. Confidence and INSUFFICIENT_DATA

`confidence = data completeness x history depth`.

- Completeness: weighted share of present fields. Stage 1: payment .25, value .15, SKU .15,
  category .15, pincode .30. Stage 2 adds `delivered_at` .25 and return reason .10 (other weights
  scaled).
- Depth: `0.5 + 0.5 x min(1, prior_orders / 5)`. A customer with no history still gets 0.5 because
  SKU, category, pincode and payment signals carry the score (cold start).
- When `delivered_at` is missing at Stage 2, confidence is capped at 0.25 because TTR is the core
  evidence.
- `INSUFFICIENT_DATA` only when confidence < 0.3. Then `risk_score`, `components` and `weights`
  are null.

## 5. Score calibration

| Component | Mapping to 0-100 |
|---|---|
| `rule` | `min(sum of points, 100)` |
| `ml` | LightGBM probability, calibrated with isotonic regression on the validation slice, x 100 |
| `anomaly` | IsolationForest score, percentile-ranked within the seller's own history (0 most normal, 100 most unusual). Off by default (`Settings.anomaly_enabled`). |

Blend: `risk_score = sum(w_i x component_i) / sum(w_i)` over present components, rounded, clipped to
0-100. Defaults: rule 0.4, ml 0.4, anomaly 0.2. Rules only gives rule 1.0. Rules plus ML gives
0.5 / 0.5. Bands: MEDIUM at 40 or more, HIGH at 70 or more (seller-editable).

The band table (share of orders and observed rate per band, against the overall rate) is produced by
`calibrate.band_table`; percentages are apportioned with largest remainder so they sum to exactly
100.0. Measured tables are in `docs/PROOF.md`.

## 6. Action matrix

| Stage | LOW | MEDIUM | HIGH | INSUFFICIENT_DATA |
|---|---|---|---|---|
| PRE_DISPATCH | NORMAL_FULFILMENT | OTP_VERIFICATION, UPI_PREPAID_INCENTIVE, PARTIAL_PREPAID | OTP_VERIFICATION, PREPAID_ONLY*, HOLD_FULFILMENT* | NORMAL_FULFILMENT |
| POST_DELIVERY | INSTANT_RETURN | PHOTO_VERIFICATION, EXCHANGE_OR_STORE_CREDIT | MANUAL_REVIEW, REFUND_AFTER_QC* | INSTANT_RETURN |

\* critical, needs approval. In OBSERVATION mode critical actions are never recommended. The first
item is `recommended_action`; the rest are `alternative_actions`. A Stage 2 MEDIUM or HIGH flag feeds
`FLAGGED_PRIOR_RETURN` into the customer's next Stage 1 assessment.

## 7. Worked examples (seed 42, n = 1,200)

**Scenario 2: COD order, customer with 2 earlier RTOs (Stage 1) gives 92, HIGH.**
40 (`COD_WITH_PRIOR_RTO`) + 20 (`RTO_RATE_HIGH`) + 14.3 (pincode lift 1.96) + 10 (2 failed
deliveries) + 6 (category lift 1.52) + 2.2 (SKU lift 1.50) = 92.5, shown as 92. Recommended:
OTP_VERIFICATION, alternatives PREPAID_ONLY and HOLD_FULFILMENT.

**Scenario 4: repeat fast returner (Stage 2) gives 87, HIGH.**
25 (3 returns in 4 earlier orders) + 20 (returned in 4.5 h against a 17.4 h seller median) + 15
(every earlier return under 6 h) + 10 (2 returns in 30 days) + 10 (3 earlier same-category
returns) + 6 (category lift) + 0.7 (SKU lift) = 86.7. Recommended: MANUAL_REVIEW, alternative
REFUND_AFTER_QC.

**Scenario 3: DEFECTIVE return in 3 h gives 0, LOW.** The reason is seller-side, so the fast TTR adds
no points and the signal `SELLER_OR_CARRIER_FAULT_REASON` is shown. Recommended: INSTANT_RETURN.

**Scenario 7: the next order by the scenario-4 customer (Stage 1) gives 72, HIGH.**
25 (4 returns in 5 earlier orders) + 15 (fast-return history) + 15 (`FLAGGED_PRIOR_RETURN`, 2
earlier flags) + 10 (recent returns) + 6 + 1.2 = 72.2. Without the earlier flag the same order
would score 57.

**Scenario 6: return with no `delivered_at` gives INSUFFICIENT_DATA.** TTR is `null`, bucket
`UNKNOWN`, confidence 0.25, `missing_fields = ["delivered_at"]`.

**Scenario 5: first-time customer, late return at 6 days gives 15, LOW.** Signals shown:
`COD_HIGH_VALUE_VS_CATEGORY` (value 2.52x the category median), `HIGH_RETURN_CATEGORY`,
`HIGH_RETURN_SKU` and `NO_PRIOR_HISTORY` (cold start). Confidence 0.5.

**Loss example (one accepted return, QC UNUSED, no cost fields supplied):**
6,000 + 8,000 + 1,500 + 2,500 = 18,000 paise (Rs 180), whether the order was Rs 1,000 or Rs 100,000.
With QC `USED` and a Rs 1,000 order the 10% write-off adds 10,000 paise.

## 8. Generator notes

`generate_dataset(n, seed)` (spec `gen-1.0.0`) plants wardrobing-style fast returners, 8 risky
pincodes, 4 high-return SKUs and low-intent COD buyers with RTO history, plus controls (genuine
defect returns, legitimate fast returns, late returns, 2% missing `delivered_at`, 1% missing
pincode, NDR and LOST). 18 fixed scenario orders (ids `ord_scn_*`, customers `cus_scn_*`) are part
of the frozen spec and count toward `n`. Hidden labels are evaluation and synthetic-training only
and are never written to CSV. The generator was frozen once the P0 tests passed and has not been
changed to make any model or test pass.
