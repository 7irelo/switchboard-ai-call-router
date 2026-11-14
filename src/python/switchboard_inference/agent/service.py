"""The Agent gRPC service: one Converse stream per call, plus a post-call summary."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import grpc
import yaml

from switchboard_inference.agent.brain import Brain, Department, Persona
from switchboard_inference.agent.listener import FluxListenConfig, FluxListener
from switchboard_inference.agent.session import AgentSession, SessionSettings
from switchboard_inference.agent.speaker import FluxSpeakConfig, FluxSpeaker, SpeakerUnavailable, synthesize
from switchboard_inference.gen.switchboard.v1 import agent_pb2, agent_pb2_grpc

log = logging.getLogger("switchboard.agent")


class GreetingCache:
    """Greetings are synthesised once, so a call is answered with no speech round trip at all."""

    def __init__(self, synth: Callable[[str, str], Any] | None):
        self.synth = synth
        self.audio: dict[tuple[str, str], bytes] = {}
        self.pending: set[tuple[str, str]] = set()
        self.tasks: set[asyncio.Task] = set()

    def get(self, voice: str, text: str) -> bytes | None:
        key = (voice, text)
        if key in self.audio:
            return self.audio[key]
        self.schedule(voice, text)  # This call speaks it live; the next one gets the cache.
        return None

    def schedule(self, voice: str, text: str) -> None:
        key = (voice, text)
        if self.synth is None or key in self.pending or key in self.audio or not text:
            return
        self.pending.add(key)
        task = asyncio.create_task(self._fill(key))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _fill(self, key: tuple[str, str]) -> None:
        try:
            for attempt in range(2):
                try:
                    self.audio[key] = await self.synth(*key)
                    break
                except SpeakerUnavailable as err:
                    if attempt:
                        log.warning("greeting not cached: %s", err)
                    else:
                        await asyncio.sleep(2)
        finally:
            self.pending.discard(key)

    def warm_from(self, tenants_dir: Path) -> int:
        count = 0
        for path in sorted(tenants_dir.glob("*.yaml")):
            agent = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("agent") or {}
            if agent.get("enabled") and agent.get("greeting"):
                for voice in {agent.get("voice", "flux-gemma-en"), *(agent.get("voice_by_extension") or {}).values()}:
                    self.schedule(voice, agent["greeting"])
                    count += 1
        return count


def persona_from(start: agent_pb2.AgentStart) -> Persona:
    return Persona(
        business_name=start.business_name,
        agent_name=start.agent_name or "the receptionist",
        greeting=start.greeting,
        knowledge=start.knowledge,
        departments=[Department(d.id, d.handles, d.open_now, d.when_closed, d.emergency) for d in start.departments],
        local_time=start.local_time,
        purpose=start.purpose,
    )


class AgentService(agent_pb2_grpc.AgentServicer):
    def __init__(
        self,
        verify: Callable[[Any], bool],
        listener_factory: Callable[[], Any] | None,
        speaker_factory: Callable[[str], Any] | None,
        brain: Brain | None,
        greetings: GreetingCache,
        summary_client: Any = None,
        settings: SessionSettings | None = None,
    ):
        self.verify = verify
        self.listener_factory = listener_factory
        self.speaker_factory = speaker_factory
        self.brain = brain
        self.greetings = greetings
        self.summary_client = summary_client
        self.settings = settings or SessionSettings()

    async def Converse(self, request_iterator, context):
        first = await anext(request_iterator, None)
        if first is None or first.WhichOneof("payload") != "start":
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "the first message must be AgentStart")
        start = first.start
        if not self.verify(start.context):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid tenant context")
        if self.listener_factory is None or self.speaker_factory is None or self.brain is None:
            # No keys: tell the gateway to use the router for this call.
            yield agent_pb2.AgentDown(end=agent_pb2.AgentEnd(action="fallback", reason="agent_not_configured"))
            return

        voice = start.voice or "flux-gemma-en"
        out: asyncio.Queue = asyncio.Queue()
        session = AgentSession(
            persona_from(start),
            self.listener_factory(),
            self.speaker_factory(voice),
            self.brain,
            out,
            greeting_audio=None if start.purpose else self.greetings.get(voice, start.greeting),
            settings=self.settings,
        )

        async def read_up() -> None:
            async for message in request_iterator:
                kind = message.WhichOneof("payload")
                if kind == "audio":
                    await session.on_audio(message.audio.ulaw)
                elif kind == "tool_result":
                    session.on_tool_result(message.tool_result.call_id, message.tool_result.ok, message.tool_result.content)
                elif kind == "mark_played":
                    session.on_mark_played(message.mark_played.id)

        run = asyncio.create_task(session.run())
        up = asyncio.create_task(read_up())
        try:
            while True:
                getter = asyncio.ensure_future(out.get())
                done, _ = await asyncio.wait({getter, run, up}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    message = getter.result()
                    yield message
                    if message.WhichOneof("payload") == "end":
                        return
                    continue
                getter.cancel()
                if up in done:
                    return  # The gateway closed the call.
                while not out.empty():
                    message = out.get_nowait()
                    yield message
                    if message.WhichOneof("payload") == "end":
                        return
                return
        finally:
            for task in (run, up):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await session.close()

    async def Summarize(self, request, context):
        if not self.verify(request.context):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid tenant context")
        if self.brain is None or not request.lines:
            return agent_pb2.SummarizeResponse()
        try:
            text, outcome = await self.brain.summarize(
                request.business_name, [(line.role, line.text) for line in request.lines], request.ending
            )
        except Exception as err:
            log.warning("summary failed: %s", type(err).__name__)
            return agent_pb2.SummarizeResponse()
        return agent_pb2.SummarizeResponse(summary=text, outcome=outcome)


def choose_brain(env: dict[str, str]) -> Brain | None:
    """AGENT_LLM picks the provider: "anthropic" or "openai". Unset: whichever has a key, Anthropic first."""
    choice = (env.get("AGENT_LLM") or "").lower()
    if not choice:
        choice = "anthropic" if env.get("ANTHROPIC_API_KEY") else "openai" if env.get("OPENAI_API_KEY") else ""
    if choice == "openai" and env.get("OPENAI_API_KEY"):
        from switchboard_inference.agent.openai_brain import DEFAULT_MODEL, OpenAIBrain

        return OpenAIBrain(model=env.get("OPENAI_AGENT_MODEL") or DEFAULT_MODEL, api_key=env["OPENAI_API_KEY"])
    if choice == "anthropic" and env.get("ANTHROPIC_API_KEY"):
        from switchboard_inference.agent.brain import ClaudeBrain

        return ClaudeBrain()
    return None


def build_agent_service(verify: Callable[[Any], bool], deepgram_key: str | None, brain: Brain | None, host: str, tenants_dir: Path | None):
    """Wires the live providers when keys exist. Without them, every call falls back to the router."""
    listener_factory = speaker_factory = None
    synth = None
    kokoro = os.environ.get("KOKORO_URL") or None
    if deepgram_key:

        def speak_config(voice: str) -> FluxSpeakConfig:
            # kokoro-* voices run locally; everything else is Deepgram Flux.
            local = kokoro if voice.startswith("kokoro-") else None
            return FluxSpeakConfig(api_key=deepgram_key, voice=voice, host=host, base_url=local)

        def listener_factory() -> FluxListener:
            return FluxListener(FluxListenConfig(api_key=deepgram_key, host=host))

        eleven_key = os.environ.get("ELEVENLABS_API_KEY") or ""

        def eleven_config(voice: str):
            from switchboard_inference.agent.eleven import ElevenConfig

            return ElevenConfig(api_key=eleven_key, voice=voice, model=os.environ.get("ELEVENLABS_MODEL") or "eleven_flash_v2_5")

        def speaker_factory(voice: str):
            if voice.startswith("eleven-"):
                from switchboard_inference.agent.eleven import ElevenSpeaker

                return ElevenSpeaker(eleven_config(voice))
            return FluxSpeaker(speak_config(voice))

        async def synth(voice: str, text: str) -> bytes:
            if voice.startswith("eleven-"):
                from switchboard_inference.agent.eleven import synthesize_eleven

                return await synthesize_eleven(eleven_config(voice), text)
            # A local CPU voice renders greetings one after another at startup, so it gets longer.
            return await synthesize(speak_config(voice), text, timeout_s=90.0 if voice.startswith("kokoro-") else 15.0)

    greetings = GreetingCache(synth)
    service = AgentService(verify, listener_factory, speaker_factory, brain, greetings)
    return service, greetings, tenants_dir
