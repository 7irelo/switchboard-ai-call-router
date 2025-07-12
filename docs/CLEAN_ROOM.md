# Clean-room record

**Status: DRAFT. G1 and G2 are not signed off.** The repo owner must complete them before anything is published.

| Gate | Requirement | Status |
|---|---|---|
| G1 | Employment contract's IP, moonlighting and confidentiality clauses read; written clearance if in doubt | **Open: owner** |
| G2 | Personal accounts, keys and git identity only; no employer credentials, VPN, cloud or vendor accounts; this laptop confirmed not employer-owned; the AI coding tool account confirmed acceptable | **Open: owner** |
| G3 | This document reviewed and signed off | **Open: owner** |

Signed off by: ________  Date: ________

## Never in this repo

Employer code, configs, dialplans, prompts, customer names, department structures, intent lists, call data, transcripts, thresholds or internal metrics.

## Where each design idea comes from

Everything below comes from public documentation or general engineering practice.

| Idea | Public source |
|---|---|
| Asterisk `chan_websocket` media path, ARI | [Asterisk docs](https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/) |
| Endpointing, interim results, `UtteranceEnd` | [Deepgram docs](https://developers.deepgram.com/docs/understand-endpointing-interim-results) |
| Expected-cost decisions | Bayes decision theory (Duda, Hart and Stork, *Pattern Classification*, ch. 2) |
| Temperature scaling | Guo et al., *On Calibration of Modern Neural Networks* (2017) |
| Conformal prediction sets | Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction* (2021) |
| PostgreSQL row-level security | [PostgreSQL docs](https://www.postgresql.org/docs/current/ddl-rowsecurity.html) |
| Outbox pattern | Common distributed-systems practice |
| Webhook signature, retry and duplicate behaviour | Public WhatsApp Cloud API documentation |
| Canonical intent names | Generic contact-centre vocabulary, written from scratch for fictional businesses |

## Synthetic-only rules

- All tenants are fictional. All utterances are generated from seeded specs in `src/python/switchboard_inference/data/`.
- Cost matrix values, rubric text and thresholds are written here from first principles or fitted on synthetic data. None is tuned against memory of any production system.
- `scripts/check_clean_room.py` scans tracked files for a locally held deny-list (`.clean-room-denylist`, gitignored). The owner adds employer and customer terms to that file; the terms themselves are never committed.
