"""The receptionist on OpenAI Realtime: one model hears the caller's audio and answers in its own voice.

It replaces Flux STT, the text model and the TTS for a call, behind the same Converse stream as AgentSession, so
the gateway is unchanged: audio, playback marks, flushes, transcripts (which still go through the emergency
guard), tool calls checked against the tenant, and the end of the call. Audio stays 8 kHz mu-law both ways.

Turn-taking is the model's semantic VAD: in a live probe it waited through a mid-sentence pause that silence
detection replied into, and still started speaking about 1.6 s after the caller finished.
"""

from __future__ import annotations

import asyncio
import base64
import bisect
import contextlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import websockets

from switchboard_inference.agent.brain import TERMINAL_TOOLS, TOOLS, Persona, build_system_prompt, dumps
from switchboard_inference.agent.session import FRAME, MARK_EVERY, FrameAssembler, SessionSettings
from switchboard_inference.gen.switchboard.v1 import agent_pb2

log = logging.getLogger("switchboard.agent")
PREFIX = "realtime-"
DEFAULT_MODEL = "gpt-realtime-mini"
REALTIME_TOOLS = [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["input_schema"]} for t in TOOLS]

SPOKEN_NOTE = """

You are speaking, not writing: your voice is what the caller hears. Sound like a calm, capable person on the phone, at a natural pace. Say prices and times naturally, like "four hundred and fifty rand" or "half past seven", taking the numbers exactly from the business facts."""


@dataclass
class _Reply:
    """One model response being played to the caller."""

    key: str
    item_id: str = ""
    sent: int = 0
    played: int = 0
    text: str = ""
    finished: bool = False  # All its audio has been sent.
    played_all: bool = False
    tools: list[dict[str, Any]] = field(default_factory=list)


class RealtimeSession:
    def __init__(
        self,
        persona: Persona,
        out: asyncio.Queue,
        api_key: str,
        voice: str,
        model: str = DEFAULT_MODEL,
        settings: SessionSettings | None = None,
        connect: Callable | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.persona = persona
        self.out = out
        self.api_key = api_key
        self.voice = voice.removeprefix(PREFIX) or "marin"
        self.model = model
        self.settings = settings or SessionSettings()
        self.connect = connect or self._connect
        self.clock = clock
        self.socket = None
        self.ended = asyncio.Event()
        self.replies: dict[str, _Reply] = {}
        self.current: _Reply | None = None
        self.assembler = FrameAssembler()
        self.tool_waiters: dict[str, asyncio.Future] = {}
        self.pending_end: tuple[str, str, str] | None = None
        self.tasks: set[asyncio.Task] = set()
        self.started = clock()
        self.turns = 0
        # (bytes received, when) for the caller's audio: maps the model's audio_end_ms to wall time.
        self.received = 0
        self.offsets: list[int] = []
        self.arrivals: list[float] = []
        self.last_word_wall: float | None = None
        self.early: list[bytes] = []
        self.first_audio_pending: float | None = None  # The caller's last word, until the reply's first audio.

    # ----- connection -----

    async def _connect(self):
        return await websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={self.model}",
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=2**24,
        )

    async def _send(self, event: dict[str, Any]) -> None:
        if self.socket is not None:
            with contextlib.suppress(websockets.WebSocketException):
                await self.socket.send(json.dumps(event))

    def _emit(self, **payload) -> None:
        self.out.put_nowait(agent_pb2.AgentDown(**payload))

    def _end(self, action: str, department: str = "", reason: str = "") -> None:
        if not self.ended.is_set():
            self._emit(end=agent_pb2.AgentEnd(action=action, department_id=department, reason=reason))
            self.ended.set()

    # ----- inbound from the gateway -----

    async def on_audio(self, ulaw: bytes) -> None:
        self.received += len(ulaw)
        self.offsets.append(self.received)
        self.arrivals.append(self.clock())
        if self.socket is None:
            self.early.append(ulaw)  # The caller spoke while connecting.
            return
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(ulaw).decode()})

    def on_tool_result(self, call_id: str, ok: bool, content: str) -> None:
        waiter = self.tool_waiters.pop(call_id, None)
        if waiter is not None and not waiter.done():
            waiter.set_result((ok, content))

    def on_mark_played(self, mark: str) -> None:
        key, _, position = mark.rpartition("|")
        reply = self.replies.get(key)
        if reply is None:
            return
        if position == "end":
            reply.played = reply.sent
            reply.played_all = True
            if reply.text:
                self._emit(transcript=agent_pb2.AgentTranscript(role="agent", text=reply.text))
            if self.pending_end and all(r.played_all for r in self.replies.values()):
                self._end(*self.pending_end)
        else:
            reply.played = max(reply.played, int(position))

    # ----- the call -----

    async def run(self) -> None:
        try:
            self.socket = await asyncio.wait_for(self.connect(), 6)
        except Exception as err:
            log.warning("realtime could not start: %s", type(err).__name__)
            self._end("fallback", reason=f"start_failed:{type(err).__name__}")
            return
        prompt = build_system_prompt(self.persona) + SPOKEN_NOTE
        await self._send({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": prompt,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcmu"},
                        "transcription": {"model": "gpt-4o-mini-transcribe"},
                        "turn_detection": {"type": "semantic_vad", "eagerness": "high", "create_response": True, "interrupt_response": True},
                    },
                    "output": {"format": {"type": "audio/pcmu"}, "voice": self.voice},
                },
                "tools": REALTIME_TOOLS,
                "tool_choice": "auto",
            },
        })  # fmt: skip
        # It opens the call itself, in its own voice: the greeting, or why it's calling.
        opening = (
            f"Start the call: introduce yourself and say why you're calling: {self.persona.purpose}"
            if self.persona.purpose
            else f'Answer the call by saying exactly: "{self.persona.greeting}"'
        )
        await self._send({"type": "response.create", "response": {"instructions": opening}})
        for chunk in self.early:
            await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode()})
        self.early.clear()
        reader = asyncio.create_task(self._read())
        limit = asyncio.create_task(self._limit())
        done, _ = await asyncio.wait([reader, limit, asyncio.create_task(self.ended.wait())], return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task is reader and task.exception() is not None:
                log.warning("realtime stopped: %s", type(task.exception()).__name__)
                self._end("fallback" if not self.replies else "error", reason=type(task.exception()).__name__)
        if not self.ended.is_set():
            self._end("error", reason="realtime_closed")

    async def _limit(self) -> None:
        await asyncio.sleep(self.settings.max_call_s)
        self.pending_end = ("transfer", "", "handoff")
        await self._send({"type": "response.create", "response": {"instructions": "Tell the caller in one sentence you're putting them through to someone who can help further.", "tool_choice": "none"}})

    async def close(self) -> None:
        for task in list(self.tasks):
            task.cancel()
        if self.socket is not None:
            with contextlib.suppress(Exception):
                await self.socket.close()

    # ----- events from OpenAI -----

    async def _read(self) -> None:
        async for raw in self.socket:
            event = json.loads(raw)
            kind = event.get("type", "")
            if kind == "response.created":
                self._new_reply(event["response"]["id"])
            elif kind == "response.output_audio.delta":
                self._audio(event)
            elif kind == "response.output_audio_transcript.done":
                reply = self.replies.get(event.get("response_id", ""))
                if reply is not None:
                    reply.text = (reply.text + " " + event.get("transcript", "")).strip()
            elif kind == "input_audio_buffer.speech_started":
                self._barge_in()
            elif kind == "input_audio_buffer.speech_stopped":
                self.last_word_wall = self._arrived_at(event.get("audio_end_ms", 0) / 1000)
            elif kind == "conversation.item.input_audio_transcription.completed":
                text = (event.get("transcript") or "").strip()
                if text:
                    self.turns += 1
                    self._emit(transcript=agent_pb2.AgentTranscript(role="caller", text=text))
            elif kind == "response.done":
                await self._response_done(event["response"])
            elif kind == "error":
                error = event.get("error", {})
                if error.get("code") not in ("response_cancel_not_active", "conversation_already_has_active_response"):
                    log.warning("realtime error: %s %s", error.get("code"), (error.get("message") or "")[:160])

    def _new_reply(self, response_id: str) -> None:
        reply = _Reply(key=response_id)
        self.replies[response_id] = reply
        if self.current is not None and not self.current.finished:
            self._finish_audio(self.current)
        self.current = reply
        self.assembler = FrameAssembler()
        self.first_audio_pending = self.last_word_wall

    def _audio(self, event: dict[str, Any]) -> None:
        reply = self.replies.get(event.get("response_id", ""))
        if reply is None or reply is not self.current:
            return  # From a response that was cut off.
        reply.item_id = event.get("item_id", reply.item_id)
        data = self.assembler.push(base64.b64decode(event["delta"]))
        if reply.sent == 0 and data and self.first_audio_pending is not None:
            first = self.clock()
            self._emit(latency=agent_pb2.TurnLatency(voice_to_voice_ms=(first - self.first_audio_pending) * 1000))
            self.first_audio_pending = None
        for start in range(0, len(data), 16000):
            piece = data[start : start + 16000]
            before = reply.sent
            reply.sent += len(piece)
            self._emit(audio=agent_pb2.AgentAudio(ulaw=piece))
            if reply.sent // MARK_EVERY != before // MARK_EVERY:
                self._emit(mark=agent_pb2.AgentMark(id=f"{reply.key}|{reply.sent}"))

    def _finish_audio(self, reply: _Reply) -> None:
        tail = self.assembler.finish()
        if tail:
            reply.sent += len(tail)
            self._emit(audio=agent_pb2.AgentAudio(ulaw=tail))
        reply.finished = True
        self._emit(mark=agent_pb2.AgentMark(id=f"{reply.key}|end"))

    def _barge_in(self) -> None:
        """The caller started talking. The model has already stopped its response; stop what's queued too."""
        reply = self.current
        if reply is None or reply.played_all or reply.sent == 0:
            return
        self._emit(flush=agent_pb2.AgentFlush())
        heard_ms = int(reply.played / 8)  # mu-law at 8 kHz: 8 bytes a millisecond.
        if reply.item_id:
            # Tell the model how much of its answer the caller actually heard.
            self._spawn(self._send({"type": "conversation.item.truncate", "item_id": reply.item_id, "content_index": 0, "audio_end_ms": heard_ms}))
        self._emit(transcript=agent_pb2.AgentTranscript(role="agent", text="", interrupted=True))
        reply.played_all = True
        reply.finished = True
        self.current = None
        if self.pending_end and self.pending_end[2] != "handoff":
            self.pending_end = None

    async def _response_done(self, response: dict[str, Any]) -> None:
        reply = self.replies.get(response.get("id", ""))
        if reply is not None and not reply.finished:
            self._finish_audio(reply)
        if response.get("status") == "failed":
            log.warning("realtime response failed: %s", json.dumps(response.get("status_details"))[:200])
        calls = [o for o in response.get("output") or [] if o.get("type") == "function_call"]
        if not calls:
            if self.pending_end and all(r.played_all for r in self.replies.values()):
                self._end(*self.pending_end)
            return
        spoke = bool(reply and reply.sent)
        terminal = False
        for call in calls:
            ok, content = await self._run_tool(call)
            await self._send({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": call["call_id"], "output": content}})
            terminal = terminal or (ok and call.get("name") in TERMINAL_TOOLS)
        if terminal:
            if not spoke:
                line = "Say one short goodbye." if self.pending_end and self.pending_end[0] == "hangup" else "Say in one short sentence who you're putting them through to."
                await self._send({"type": "response.create", "response": {"instructions": line, "tool_choice": "none"}})
            elif all(r.played_all for r in self.replies.values()):
                self._end(*self.pending_end)
        else:
            await self._send({"type": "response.create"})

    async def _run_tool(self, call: dict[str, Any]) -> tuple[bool, str]:
        call_id = call["call_id"]
        name = call.get("name", "")
        try:
            arguments = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        waiter = asyncio.get_running_loop().create_future()
        self.tool_waiters[call_id] = waiter
        self._emit(tool_call=agent_pb2.ToolCall(call_id=call_id, name=name, arguments_json=dumps(arguments)))
        try:
            ok, content = await asyncio.wait_for(waiter, self.settings.tool_timeout_s)
        except TimeoutError:
            return False, dumps({"error": "the system didn't respond in time"})
        if ok and name == "transfer_to_department":
            self.pending_end = ("transfer", str(arguments.get("department", "")), str(arguments.get("reason", ""))[:200])
        elif ok and name == "end_call":
            self.pending_end = ("hangup", "", str(arguments.get("reason", ""))[:200])
        return ok, content

    def _arrived_at(self, seconds: float) -> float | None:
        index = bisect.bisect_left(self.offsets, int(seconds * 8000))
        return self.arrivals[index] if index < len(self.arrivals) else None

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)


__all__ = ["DEFAULT_MODEL", "FRAME", "PREFIX", "RealtimeSession"]
