# Failure log

Real failures hit while building this, with what they taught. Each entry links to its fix.

## 1. The expected-cost rule sent possible emergencies to voicemail

**Found by:** property tests in both engines, on the first run. Python's Hypothesis shrank it to Thornveld at night, 6.25% emergency, 93.75% billing, clarify turns used up.

**Cause:** the design was right about the maths and wrong about the goal. Voicemail costs 100p and a false emergency route costs 8(1-p), so voicemail wins below about 7.4%. The metric says zero emergencies to voicemail, and a matrix alone can't promise zero.

**Fix:** a hard guard at 2% (docs/adr/0002). **Lesson:** a cost matrix expresses preferences, and an invariant needs a rule.

## 2. A lexicon hit could still get a clarifying question

**Found by:** the same run. A caller saying "not breathing" with inference down was asked to clarify, because clarify's expected cost (2.75) beat a route to on-call (4).

**Fix:** a lexicon hit routes directly. **Lesson:** the safety net has to be deterministic, or it isn't one.

## 3. The isolation suite passed with `SET LOCAL` removed

**Found by:** the mutation check. Replacing `SET LOCAL` with a session-level `SET` left every test green, because Npgsql resets connections on return to the pool and hid the bug.

**Why it matters:** anyone later enabling `No Reset On Close` for speed would have leaked one tenant's identity onto the next transaction on that connection, with no test failing.

**Fix:** the pool test now turns that reset off, so only `SET LOCAL` can pass. **Lesson:** a green isolation suite proves little until it has been seen to go red.

## 4. Bridging before the media channel was ready

**Found by:** the first concurrent SIP calls. ARI's `externalMedia` returns before the channel enters Stasis, so `addChannel` sometimes got a 422, and 3 of 5 calls fell back to reception.

**Fix:** wait for the channel's `StasisStart` event before bridging. **Lesson:** the safe fallback did its job, and a caller never knew.

## 5. A passing test that shouldn't have

**Found by:** reading the logs behind a mostly green run. Two of those failed calls reached the expected phone anyway, because those tenants' fallback happens to be the same department.

**Fix:** E2E now also requires a `routed` decision for each tenant, and no fallback in the logs. **Lesson:** check why something ended up in the right place, as well as that it did.

## 6. Concurrent test callers hung on BYE

**Found by:** E2E, only with concurrent calls. Every SIPp container runs as the same PID, so they all produced the same SIP branch. Asterisk treated one caller's BYE as a retransmission of another's and never answered it.

**Fix:** a per-call salt in the branch. This was a harness bug: real phones generate unique branches, as RFC 3261 requires.

## 7. The worker died when its dependencies came back

**Found by:** reading container state after failure injection had reported success. The scenarios passed, because calls did route, but the event worker had exited. An unhandled exception in a .NET hosted service stops the host, with exit code 0.

**Fix:** the worker now retries on any failure, Compose restarts it, the gateway's publisher handles timeouts, and failure injection fails if any service is left dead. **Lesson:** "the call worked" isn't the same as "the system recovered".

## 8. The gate couldn't see a shifted threshold

**Found by:** the gate's self-test. Moving both thresholds to 1000 left it green, because TF-IDF is near-perfect on template fixtures, so the thresholds never come into play.

**Status: open.** The self-test now seeds broken labels and garbled input, which do turn it red. Threshold regressions can only be tested on harder, real data. **Lesson:** a gate is only as sensitive as the data behind it.

## 9. Calls stayed degraded after inference had recovered

**Found by:** failure injection. A "Postgres down" call reached no department, and the logs showed Postgres wasn't the problem: the config cache was fine, but inference had been restarted by the previous scenario, and gRPC was still backing off.

**Cause:** the client's default reconnect backoff can reach 2 minutes. Inference could be healthy while every caller still got the degraded path.

**Fix:** backoff capped at 3 seconds, and scenarios now wait for each dependency to be healthy. **Lesson:** an outage ends when callers stop feeling it, not when the service comes back.
