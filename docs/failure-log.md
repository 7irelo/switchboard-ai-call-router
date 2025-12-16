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

## 10. Load failures were RTP port exhaustion

**Found by:** load testing. A quarter of calls failed at 4 calls per second. The gateway looked innocent, and Asterisk's log said it couldn't allocate RTP ports.

**Cause:** a 100-port range, chosen for softphone NAT, gave about 50 concurrent streams. Each routed call uses several.

**Fix:** a wider range, since no RTP ports are published. **Lesson:** the first saturation point was a config value, not code, and only a load test would have shown it.

## 11. Keyword scoring failed on the first real transcript

**Found by:** the first real speech through Asterisk into Whisper, which heard "My rooter keeps retarting my self and I need help". The keyword model found nothing, and the caller was asked to clarify.

**Fix:** serving uses char-n-gram TF-IDF, which tolerates "rooter". **Lesson:** every test before this had used clean scripted text. The first real audio broke something within seconds, and this is why the plan insists on measured noise.

## 12. A quiet caller was marked as an inference outage

**Found by:** the same call. A turn timing out cancels its gRPC stream, and the gateway logged that as "inference unavailable", flipping the call into degraded mode.

**Fix:** a cancelled stream on a timed-out turn is now just a timeout. **Lesson:** telling apart "nobody spoke" and "the system is broken" has to be explicit. Both look like silence.

## 13. One good latency number, and a hang-up treated as a failure

**Found by:** sampling repeatedly instead of quoting one call. Approximate time to route had been about 1.6 s, then about 8 s on identical code, with 2 of 8 calls never routed.

**Cause:** the host was at 99% CPU from other work, which took Whisper from about 1 s to as much as 13 s. Whisper then outran the turn timeout, the caller hung up, and the gateway treated that cancelled stream as a failure and tried to route a call that no longer existed.

**Fix:** a hang-up is now a normal ending, and Whisper makes one pass with bounded threads and a longer turn timeout. The latency itself isn't fixed, only reported. **Lesson:** the 1.6 s figure would have read well, and it was luck.

## 14. The first CI run failed on a race I'd been hiding

**Found by:** GitHub Actions, on a clean Linux runner. Every call was hung up on, because E2E started within seconds of seeding, before the gateway's 3-second config refresh had seen the new tenants.

**Why it passed locally:** I'd been pausing between steps by hand, so the race never showed. Anyone following the README would have hit it.

**Fix:** `/healthz` reports how many tenants are loaded, and E2E waits for all of them. **Lesson:** a clean machine doesn't share your habits.
