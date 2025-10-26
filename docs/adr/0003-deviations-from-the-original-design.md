# ADR 0003: Where the build differs from the original design

Status: accepted.

| Original intent | What was built | Why |
|---|---|---|
| Redis Streams plus a Postgres outbox | Redis Streams only, with a bounded in-memory queue in front | The hot path must never touch the database. Events can drop under a Redis outage, with a counter. That's acceptable for analytics events and is stated in the failure table. It wouldn't be for billing. |
| Separate Langfuse project per tenant | Per-tenant trace sinks proven at the Collector; Langfuse itself uses one project with tenant tags | Headless Langfuse init creates one project. This is the documented cut-list fallback, and it's a weaker boundary for anyone with access to that project. |
| Lexicon sets a probability floor | A lexicon hit routes directly | See ADR 0002. |
| Clarifying question names the two candidates, in voice | Generic spoken clarify prompt; candidates are named only in text messages | Prompts are pre-rendered and fixed, so naming pairs would need a clip per pair. |
| Local VAD estimates `t_eos` on live calls | Present in the Deepgram and Whisper adapters | Its error against true end of speech hasn't been measured on live calls. |
| South African TTS voices | espeak-ng as a stand-in | No TTS account is configured. isiZulu prompts are spoken with an English voice until a licensed voice replaces them. |
