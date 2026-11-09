<div align="center">

<img src="public/returniq-logo.png" alt="ReturnIQ logo" width="120" />

# ReturnIQ

**Don't manage the return. Prevent the next one.**

Explainable return-prevention intelligence for e-commerce sellers: predict COD/RTO failures and post-delivery return abuse before the loss happens.

[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7-02A676)](https://lightgbm.readthedocs.io/)
[![Cloudflare Workers](https://img.shields.io/badge/Cloudflare-Workers-F38020?logo=cloudflare&logoColor=white)](https://workers.cloudflare.com/)
[![Drizzle ORM](https://img.shields.io/badge/Drizzle-ORM-C5F74F?logo=drizzle&logoColor=black)](https://orm.drizzle.team/)
[![pnpm](https://img.shields.io/badge/pnpm-11-F69220?logo=pnpm&logoColor=white)](https://pnpm.io/)
[![Built by TAPWEBS](https://img.shields.io/badge/Built_by-TAPWEBS-111827)](https://github.com/tapwebs)

</div>

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Format](#data-format)
- [Risk Engine](#risk-engine)
- [Metrics](#metrics)
- [Roadmap](#roadmap)
- [Team](#team)
- [License](#license)

---

## Overview

ReturnIQ is a lightweight intelligence layer that sits on top of a seller's existing commerce data. It reads orders, shipments and returns, scores risk at two points in the order lifecycle, and recommends a targeted intervention for each flagged case. It does not apply blanket restrictions.

Most tools report losses after they happen. ReturnIQ scores the order **before dispatch** and the return **before the refund**, and explains every score.

## The Problem

Sellers lose money to:

- COD failures and Return-to-Origin (RTO)
- Post-delivery return abuse, including wardrobing
- Repeat return offenders and low-intent buyers
- Forward and reverse shipping, packaging and handling costs
- Blocked inventory and delayed cash flow

## Key Features

| Feature | Description |
|---|---|
| **Dual-stage risk assessment** | Stage 1 scores an order at dispatch (COD, RTO, low-intent). Stage 2 scores a return request before refund (abuse, wardrobing, repeat returns). |
| **Explainable scores** | Each assessment has a 0-100 risk index, a band, a confidence value, the triggered signals and a suggested action. |
| **Return DNA** | Seller-specific behavior profile built from customer, SKU, payment mode, pincode, delivery outcome, return history and time-to-return. |
| **Hybrid engine** | A transparent rules engine combined with a LightGBM model and anomaly detection. |
| **Action engine** | Approval-driven interventions: OTP verification, partial prepaid, seller review, photo verification, refund after QC. |
| **Loss estimation** | Observed operating loss and scenario-based avoidable loss (low / base / high). Assumptions are labelled. |
| **Policy simulation** | Preview the effect of a policy before applying it. |
| **Leakage-safe features** | Every feature is computed strictly before the `as_of` time of the stage being scored. |

Example assessment:

```text
Risk index: 87 / 100  (HIGH)

Signals:
  - 4 returns in 5 orders
  - Abnormally short return timing
  - Repeat SKU category
  - Previous COD failures
  - High-risk pincode cluster
```

## How It Works

```text
ORDER -> DELIVERY -> RETURN -> LOSS -> LEARNING -> NEXT ORDER
```

1. **Upload** `orders.csv`, plus returns and shipments data.
2. **Analyse.** ReturnIQ computes RTO rate, return rate, COD vs prepaid trends, repeat-return behavior, risky pincodes and SKUs, return reasons and time-to-return.
3. **Act.** Review flagged orders and returns, approve or reject suggested interventions, and feed outcomes back into future scoring.

## Architecture

```text
┌──────────────────────────┐
│  Data connectors         │  CSV upload, demo stores
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  Normalisation           │  prepare.py, validation
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  Hybrid risk engine      │  rules + LightGBM + anomaly
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  Action engine           │  recommend, approve, track
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  Workspace UI            │  Next.js on Cloudflare Workers
└──────────────────────────┘
```

The Python engine is pure: no I/O, no HTTP, no database. The web app talks to it through a versioned shared contract (`backend/packages/shared`).

## Tech Stack

| Layer | Technology |
|---|---|
| Web app | Next.js 16 (via vinext), React 19, TypeScript, Tailwind CSS 4 |
| UI | shadcn/ui, Radix UI, Base UI, Motion, Recharts, Sonner |
| Data | Drizzle ORM, Cloudflare D1 (SQLite) |
| Runtime | Cloudflare Workers, Vite 8, Wrangler |
| Risk engine | Python 3.11+, pandas, NumPy, scikit-learn, LightGBM, Pydantic |
| Tooling | pnpm, ESLint, pytest, Ruff |

## Project Structure

```text
ReturnIQ/
├── app/                  # Next.js routes, workspace UI, API route
├── components/           # Shared UI components
├── lib/                  # Connectors, demo stores, workspace logic
├── db/                   # Drizzle schema
├── drizzle/              # Migrations
├── public/               # Static assets, orders-template.csv
├── scripts/              # Build and install helpers
└── backend/
    ├── docs/             # metrics.md, PROOF.md
    └── packages/
        ├── engine/       # returniq_engine (rules, ML, analytics, tests)
        └── shared/       # Versioned contract and JSON fixtures
```

## Getting Started

### Prerequisites

- Node.js >= 22.13.0
- pnpm 11
- Python >= 3.11 (only for the risk engine)

### Web app

```bash
pnpm install
pnpm dev
```

Other scripts:

```bash
pnpm build     # production build
pnpm start     # run the built worker locally with Wrangler
pnpm lint      # ESLint
```

### Risk engine

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e "packages/engine[dev]"
pytest packages/engine -q
```

Generate the synthetic report:

```bash
python -m returniq_engine.report --seed 42 --eval --perf
```

## Data Format

A starter file is available at [`public/orders-template.csv`](public/orders-template.csv).

| Column | Example |
|---|---|
| `order_id` | `ord_example_1` |
| `customer_id` | `cus_example_1` |
| `order_date` | `2026-10-04T10:00:00+05:30` |
| `sku_id` | `sku_001` |
| `quantity` | `1` |
| `payment_mode` | `COD` |
| `order_amount_inr` | `1499` |
| `pincode` | `302021` |

Money is stored as integer paise inside the engine.

## Risk Engine

- **Rules:** versioned (`rules-1.0.0`), points-based, fully explainable.
- **Leakage rule:** history is computed with strict `as_of` cut-offs. Tests mutate every later event and assert the assessment does not change.
- **Seller-fault protection:** returns for defective, damaged-in-transit or wrong-item reasons are never counted as preventable.
- **Unknown stays unknown:** missing data gives `null` or `INSUFFICIENT_DATA`, never a guessed value.

## Metrics

The north-star metric is the **Preventable Return Rate (PRR)**:

```text
PRR = Preventable Returns / Delivered Orders x 100
```

ReturnIQ reports an *estimated* PRR (what the system would have flagged) and a *confirmed* PRR (seller-confirmed, `null` until confirmations exist). RTO and post-delivery returns are never combined into one number.

> All figures in [`backend/docs/PROOF.md`](backend/docs/PROOF.md) come from a **synthetic dataset with planted patterns**. They check that the engine finds known structure. They are not evidence of lift on a real seller.

Full definitions: [`backend/docs/metrics.md`](backend/docs/metrics.md).

## Roadmap

- [x] CSV ingestion and analytics dashboard
- [x] Rules engine and two-stage risk scoring
- [x] Action engine and loss estimation
- [ ] Real-time risk scoring API
- [ ] Seller pilot program
- [ ] Shopify, WooCommerce and ONDC adapters
- [ ] UPI interventions
- [ ] Multi-tenant SaaS

## Team

| Member | Role |
|---|---|
| Akshat Sharma | Tech Lead, Backend and Risk Engine |
| Tanish Khandelwal | Product Lead, Research, Strategy and UX |
| Priyanshu Mahovia | Frontend Lead, UI/UX |
| Vedansh Patel | Presentation and Business Strategy |

## License

No license has been declared yet. All rights reserved by the authors until one is added.

---

<div align="center">

**ReturnIQ: Learn. Predict. Prevent.**

Built with care by [**TAPWEBS**](https://github.com/tapwebs)

</div>
