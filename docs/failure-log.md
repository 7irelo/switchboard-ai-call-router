# Failure log

Real failures hit while building this, with what they taught. Each entry links to its fix.

## 1. The expected-cost rule sent possible emergencies to voicemail

**Found by:** property tests in both engines, on the first run. Python's Hypothesis shrank it to Thornveld at night, 6.25% emergency, 93.75% billing, clarify turns used up.

**Cause:** the design was right about the maths and wrong about the goal. Voicemail costs 100p and a false emergency route costs 8(1-p), so voicemail wins below about 7.4%. The metric says zero emergencies to voicemail, and a matrix alone can't promise zero.

**Fix:** a hard guard at 2% (docs/adr/0002). **Lesson:** a cost matrix expresses preferences, and an invariant needs a rule.

## 2. A lexicon hit could still get a clarifying question

**Found by:** the same run. A caller saying "not breathing" with inference down was asked to clarify, because clarify's expected cost (2.75) beat a route to on-call (4).

**Fix:** a lexicon hit routes directly. **Lesson:** the safety net has to be deterministic, or it isn't one.
