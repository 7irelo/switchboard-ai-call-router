"""The inference gRPC service. Python scores; .NET decides. No tenant state outlives a request."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import grpc

from switchboard_inference.asr.base import AsrUnavailable
from switchboard_inference.asr.failover import FailoverSession
from switchboard_inference.asr.replay import ReplayScripts, ReplaySession
from switchboard_inference.gen.switchboard.v1 import common_pb2, inference_pb2, inference_pb2_grpc
from switchboard_inference.intent.baselines import KeywordModel, TfidfModel
from switchboard_inference.llm.fusion import ConfusionFusion
from switchboard_inference.telemetry import LoopLagMonitor, configure_tracing, span

log = logging.getLogger("switchboard.inference")
OUTCOMES = {
    "answered": inference_pb2.ESCALATE_OUTCOME_ANSWERED,
    "needs_clarification": inference_pb2.ESCALATE_OUTCOME_NEEDS_CLARIFICATION,
    "refused": inference_pb2.ESCALATE_OUTCOME_REFUSED,
    "truncated": inference_pb2.ESCALATE_OUTCOME_TRUNCATED,
    "error": inference_pb2.ESCALATE_OUTCOME_ERROR,
    "unavailable": inference_pb2.ESCALATE_OUTCOME_UNAVAILABLE,
}


class Scorer(Protocol):
    version: str

    def predict(self, text: str) -> dict[str, float]: ...


def sign_context(secret: str, tenant_id: str, conversation_id: str, config_version: str) -> str:
    message = f"{tenant_id}|{conversation_id}|{config_version}".encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def verify_context(secret: str, ctx: common_pb2.TenantContext) -> bool:
    if not ctx.tenant_id or not ctx.signature:
        return False
    expected = sign_context(secret, ctx.tenant_id, ctx.conversation_id, ctx.config_version)
    return hmac.compare_digest(expected, ctx.signature)


def to_scores(scores: dict[str, float], allowed: list[str]) -> list[common_pb2.IntentScore]:
    """Returns only this tenant's intents, so a label from another tenant can't travel back."""
    scope = set(allowed)
    return [common_pb2.IntentScore(intent=k, probability=v) for k, v in scores.items() if k in scope]


class InferenceService(inference_pb2_grpc.InferenceServicer):
    def __init__(self, secret: str, scorer: Scorer, asr_factory, llm=None, fusion: ConfusionFusion | None = None):
        self.secret = secret
        self.scorer = scorer
        self.asr_factory = asr_factory
        self.llm = llm
        self.fusion = fusion or ConfusionFusion({})
        # One dedicated worker, so scoring can't block the event loop that handles audio.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="encoder")

    async def _score(self, text: str) -> tuple[dict[str, float], float]:
        started = time.perf_counter()
        scores = await asyncio.get_running_loop().run_in_executor(self.executor, self.scorer.predict, text)
        return scores, (time.perf_counter() - started) * 1000

    async def _require_context(self, ctx: common_pb2.TenantContext, context) -> None:
        if not verify_context(self.secret, ctx):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid tenant context")

    async def ClassifyText(self, request, context):
        await self._require_context(request.context, context)
        with span("classify_text", request.context.tenant_id, request.context.conversation_id):
            scores, ms = await self._score(request.text)
        return inference_pb2.ClassifyTextResponse(
            scores=to_scores(scores, list(request.allowed_intents)), model_version=self.scorer.version, encoder_ms=ms
        )

    async def Escalate(self, request, context):
        await self._require_context(request.context, context)
        if self.llm is None:
            return inference_pb2.EscalateResponse(outcome=inference_pb2.ESCALATE_OUTCOME_UNAVAILABLE)
        allowed = list(request.allowed_intents)
        custom = [{"id": c.id, "description": c.description, "examples": list(c.examples)} for c in request.custom_intents]
        remaining = context.time_remaining()
        with span("llm", request.context.tenant_id, request.context.conversation_id) as current:
            result = await self.llm.classify(request.transcript, allowed, request.prompt_override, custom, deadline_s=remaining)
            if current is not None:
                current.set_attribute("llm.outcome", result.outcome)
                current.set_attribute("llm.model", result.model)
                current.set_attribute("llm.cost_usd", result.cost_usd)
        response = inference_pb2.EscalateResponse(
            outcome=OUTCOMES[result.outcome],
            intent=result.intent or "",
            alternative=result.alternative or "",
            model=result.model,
            llm_ms=result.llm_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        if result.intent:
            prior = {s.intent: s.probability for s in request.encoder_scores}
            fused = self.fusion.fuse(prior, result.intent, allowed)
            response.fused_scores.extend(to_scores(fused, allowed))
        return response

    async def StreamTurn(self, request_iterator, context):
        first = await anext(request_iterator, None)
        if first is None or first.WhichOneof("payload") != "start":
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "the first message must be TurnStart")
        start = first.start
        await self._require_context(start.context, context)
        allowed = list(start.allowed_intents)

        session = await self.asr_factory(start)
        output: asyncio.Queue = asyncio.Queue()

        async def read_audio() -> None:
            try:
                async for message in request_iterator:
                    kind = message.WhichOneof("payload")
                    if kind == "audio":
                        await session.send_audio(message.audio.ulaw)
                    elif kind == "end":
                        await session.finish()
                        return
            except AsrUnavailable as err:
                log.warning("asr unavailable: %s", type(err).__name__)

        async def read_events() -> None:
            async for event in session.events():
                if event.kind == "degraded":
                    await output.put(inference_pb2.StreamTurnResponse(degraded=inference_pb2.Degraded(reason=event.source)))
                elif event.kind == "end_of_speech":
                    await output.put(inference_pb2.StreamTurnResponse(end_of_speech=inference_pb2.EndOfSpeech(source=event.source)))
                else:
                    is_final = event.kind == "final"
                    await output.put(
                        inference_pb2.StreamTurnResponse(
                            transcript=inference_pb2.TranscriptUpdate(
                                text=event.text, is_final=is_final, asr_confidence=event.confidence, language=event.language
                            )
                        )
                    )
                    if event.text:
                        scores, ms = await self._score(event.text)
                        await output.put(
                            inference_pb2.StreamTurnResponse(
                                score=inference_pb2.ScoreUpdate(
                                    scores=to_scores(scores, allowed), from_final=is_final,
                                    model_version=self.scorer.version, encoder_ms=ms,
                                )
                            )
                        )  # fmt: skip
            await output.put(None)

        audio_task = asyncio.create_task(read_audio())
        events_task = asyncio.create_task(read_events())
        try:
            with span("asr_turn", start.context.tenant_id, start.context.conversation_id):
                while (item := await output.get()) is not None:
                    yield item
        finally:
            audio_task.cancel()
            events_task.cancel()
            await session.close()


def build_asr_factory(mode: str, replay_path: str | None):
    scripts = ReplayScripts.load(replay_path) if replay_path else ReplayScripts({})
    turn_index: dict[str, int] = {}

    async def replay(start):
        # Each new turn of one conversation gets the next scripted line.
        conversation = start.context.conversation_id
        turns = scripts.turns(start.replay_key)
        index = turn_index.get(conversation, 0)
        turn_index[conversation] = index + 1
        return ReplaySession(turns[min(index, len(turns) - 1)])

    if mode == "replay":
        return replay

    if mode == "whisper":
        from switchboard_inference.asr.whisper import WhisperSession, load_whisper

        model = load_whisper(os.environ.get("WHISPER_SIZE") or "tiny")

        async def whisper_only(start):
            return WhisperSession(model)

        return whisper_only

    from switchboard_inference.asr.deepgram import DeepgramConfig, DeepgramSession

    config = DeepgramConfig(api_key=os.environ["DEEPGRAM_API_KEY"], host=os.environ.get("DEEPGRAM_HOST", "api.deepgram.com"))
    whisper_model = None
    if os.environ.get("WHISPER_SIZE"):
        from switchboard_inference.asr.whisper import load_whisper

        whisper_model = load_whisper(os.environ["WHISPER_SIZE"])

    async def live(start):
        async def primary():
            session = DeepgramSession(config)
            await session.connect()
            return session

        async def fallback():
            from switchboard_inference.asr.whisper import WhisperSession

            return WhisperSession(whisper_model)

        session = FailoverSession(primary, fallback if whisper_model else None)
        await session.start()
        return session

    return live


async def serve(args) -> None:
    secret = os.environ["INFERENCE_CONTEXT_SECRET"]
    scorer: Scorer = KeywordModel()
    if args.encoder:
        from switchboard_inference.intent.encoder import OnnxEncoder

        scorer = OnnxEncoder(args.encoder, temperature=args.temperature)
        scorer.predict("warm up before reporting ready")
    elif args.model:
        scorer = TfidfModel.load(args.model)
    tracing = configure_tracing()
    lag = asyncio.create_task(LoopLagMonitor().run())
    llm = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from switchboard_inference.llm.client import LlmClient

        llm = LlmClient(args.llm_config)
    fusion = ConfusionFusion.load(args.fusion) if args.fusion and Path(args.fusion).exists() else None
    service = InferenceService(secret, scorer, build_asr_factory(args.asr, args.replay), llm, fusion)
    server = grpc.aio.server()
    inference_pb2_grpc.add_InferenceServicer_to_server(service, server)
    server.add_insecure_port(f"[::]:{args.port}")
    await server.start()
    log.info("inference ready scorer=%s asr=%s llm=%s tracing=%s", scorer.version, args.asr, llm is not None, tracing)
    try:
        await server.wait_for_termination()
    finally:
        lag.cancel()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--asr", choices=["replay", "live", "whisper"], default="replay")
    parser.add_argument("--replay", default=None)
    parser.add_argument("--model", default=os.environ.get("INFERENCE_MODEL") or None, help="a pickled TF-IDF model")
    parser.add_argument("--encoder", default=None, help="an exported ONNX encoder directory")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--fusion", default=None)
    parser.add_argument("--llm-config", default="configs/llm_models.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}')
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
