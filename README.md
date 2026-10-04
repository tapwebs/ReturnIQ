# ReturnIQ sandbox
A private, persistent demo website with Seller and Rider modes, shared order drawers, delivery outcomes, rider scorecards, simulated customer reviews, CSV order import and sandbox approvals.

This hosted preview uses React with the Sites Vinext adapter and D1. It is not the React Router / FastAPI / Python monorepo specified in the team's implementation plan. The React workspace lives in app/workspace.tsx; sample models in lib/model.ts; server operations in app/api/workspace/route.ts. Port these interfaces to apps/web when integrating the original API.

## Implemented
- Private-site identity scopes each tenant; random HttpOnly sessions carry server-stored Seller/Rider roles.
- Rider reads filter assignments server-side, and seller operations return 403 for rider sessions.
- Durable outcomes, reviews, proposals, decisions and audit history. Optimistic concurrency protects writes.
- Decision versions prevent duplicate sandbox execution. Rejection does not execute.
- Demo role switching is explicitly a demo feature. This deployment is demo-only; do not treat its role picker as production authentication.
- Synthetic scores are fixture values, not ML; playground explicitly uses illustrative rules.

## Deferred integration
Python engine, calibrated scoring, FastAPI contract, Supabase authentication, external messaging, shipment/return CSV files, real tokenized review links, SLA timestamps, stage-2 calculations and outcome effectiveness. Public review URL deliberately omitted: hackathon uses the drawer simulation only.

Rider expected rates are transparent assumptions (6/12/20% by band); minimum 20 attempted COD deliveries. No causal rider judgement. All customer reviews in this sandbox are simulated.

## Validation
TypeScript and production build required before publishing. Browser/WebMCP validation unavailable without the permitted browser skill.

## October UI update
- Uploaded RQ logo used across landing, shell, login and icon metadata.
- Expanded landing sections, interactive Seller/Rider tour, FAQs and reduced-motion-aware Framer Motion (Motion for React).
- Three tenant-scoped demo stores. Nimbus retains its original data; Aurora and Circuit have separate durable rows.
- Five-minute in-memory view cache, concurrent read deduplication and background revalidation. No durable records moved into browser storage. Role changes clear all view caches; stale workspace responses are ignored. Writes remain server-confirmed.
- Plain status typography replaces pill/capsule visual treatments.
- Run `node tests/workspace.mjs` for the isolated SQLite-backed API/cache checks.
- Local supervised preview was started. User-facing local preview transport is unavailable in this environment; use the published site for review.
