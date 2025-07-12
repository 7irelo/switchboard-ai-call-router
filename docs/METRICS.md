# Pre-registered metrics

Pre-registration date: 2026-11-03, before any encoder training. Don't change this file or `configs/cost_matrix.yaml` after seeing test results; any later change is listed under "Post-hoc changes" and labelled in the report. Targets are hypotheses.

## Primary

| Metric | Definition | Target |
|---|---|---|
| Mean misroute cost | Mean of the cost matrix per conversation, including clarification and latency penalties | Cascade lower than **both** encoder-only and LLM-only, by paired bootstrap |
| Routing accuracy | Final destination correct, overall and per tenant | ≥ 0.90 overall; ≥ 0.85 each tenant |
| Emergency recall | Emergencies reaching the emergency route | ≥ 0.98 |
| Emergency to voicemail | Count | **0**, with its upper bound |
| Time to route (`T_route`) | True end of speech to routing decision, first turn | p95 ≤ 1,500 ms on fast and LLM paths |
| Clarification rate | Conversations with at least one clarifying turn | ≤ 15% |
| Clarification success | Correct route after a clarifying turn | ≥ 0.85 |
| Cross-tenant leaks | Canary hits, policy failures | **0, hard gate** |

## Secondary

LLM escalation rate and cost per 1k; code-switch gap (target ≤ 5 points per language pair); premature-route rate; earliness and prefix stability; ECE and risk-coverage; encoder p95 on fixed cores (≤ 30 ms); ASR WER per language; Whisper vs Deepgram routing accuracy; total time to route; sim-to-real gap.

## Latency anchors

`T_route = t_decision − t_eos`. `t_eos` is true end of speech from the synthetic audio, or the VAD estimate on live calls with its error reported. `t_route_cmd` and `t_ring` are reported separately. All times come from one host's monotonic clock. Each figure states hardware, vantage, ASR region, concurrency and sample count.

## Statistics

Wilson intervals for proportions. Stratified bootstrap (10k resamples) by conversation for composites. McNemar for correct/incorrect pairs; paired bootstrap for mean cost. Holm correction across primary comparisons. Every slice is exploratory. Oversampled sets report weighted and per-stratum estimates.

## Stop rules

- Encoder doesn't beat TF-IDF on mean cost: ship the simpler model and say so.
- LLM-only beats the cascade on cost and p95: publish that, then test a smaller model before concluding.
- isiZulu ASR unusable: narrow the claim and publish the limitation.
- Budget can't be met from this vantage: report the breakdown and run co-located. The budget doesn't change.

## Post-hoc changes

None.
