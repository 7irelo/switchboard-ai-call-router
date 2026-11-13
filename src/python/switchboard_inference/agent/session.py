"""One phone call with the receptionist agent: listen, think, speak, repeat, until a transfer or goodbye.

Where the time goes, and what this does about it:
- Turn detection: Flux decides the caller is done. On its *eager* signal the model starts at once, in held mode:
  its words are collected but not spoken and tools don't run, until EndOfTurn confirms the same words.
- Thinking: the response streams, and each piece of text goes to speech as it arrives.
- Speaking: audio goes to Asterisk as it arrives. Asterisk queues it, so the session tracks what was actually
  played using marks, and the history only ever says what the caller really heard.
- Interruptions: when the caller starts talking over the agent, its queued audio is flushed, the response is
  cancelled, and the speaker switches to a spare connection.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from switchboard_inference.agent.brain import (
    TERMINAL_TOOLS,
    Brain,
    Persona,
    Reply,
    ToolUse,
    build_system_prompt,
    caller_message,
    dumps,
    tool_result_message,
)
from switchboard_inference.agent.listener import TurnEvent
from switchboard_inference.gen.switchboard.v1 import agent_pb2

log = logging.getLogger("switchboard.agent")

FRAME = 160  # 20 ms of mu-law at 8 kHz: the size Asterisk plays without dropping anything.
SILENCE = b"\xff"
MARK_EVERY = 4000  # Bytes between playback marks: half a second.
CHARS_PER_SECOND = 14.0  # For estimating how much of an utterance played when its total length isn't known yet.
# The agent's own voice can come back through a caller's speakerphone. Words heard while it's talking, or just
# after, that mostly repeat its own are echo, not the caller. Interrupting needs this many words of the caller's own.
ECHO_WINDOW_S = 2.5
ECHO_OVERLAP = 0.6
BARGE_IN_WORDS = 2


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("\u2019", "'"))


# Filler words say nothing about whether speech is an echo: "can I book" isn't an echo of "how can I help".
FILLER = set(
    "a an and are be but can could do for have how i i'm is it it's me my no of okay on or please so that the "
    "this to uh um we what yes you your".split()
)


@dataclass
class SessionSettings:
    speculate: bool = True
    silence_s: float = 8.0
    max_turns: int = 30
    max_call_s: float = 600.0
    tool_timeout_s: float = 3.0
    max_rounds: int = 3


@dataclass
class Utterance:
    key: str
    text: str = ""
    sent: int = 0  # Bytes sent to Asterisk.
    played: int = 0  # Bytes Asterisk has confirmed playing.
    received_all: bool = False
    played_all: bool = False
    interrupted: bool = False
    finished_at: float | None = None  # When it stopped playing, finished or cut off.

    def heard(self) -> str:
        """What the caller actually heard, cut at a word boundary."""
        if self.played_all:
            return self.text
        if self.received_all and self.sent:
            chars = int(len(self.text) * self.played / self.sent)
        else:
            chars = int(self.played / 8000 * CHARS_PER_SECOND)
        if chars >= len(self.text):
            return self.text
        cut = self.text[:chars]
        return cut[: cut.rfind(" ")] if " " in cut else ""


@dataclass
class Response:
    caller_text: str
    base: list[dict[str, Any]]
    held: bool
    released: asyncio.Event = field(default_factory=asyncio.Event)
    messages: list[dict[str, Any]] = field(default_factory=list)
    streaming: bool = False  # A round's text is going out and isn't in `messages` yet.
    utterance: Utterance | None = None
    first_utterance: str = ""
    task: asyncio.Task | None = None
    started: float = 0.0
    first_token: float | None = None
    first_text_sent: float | None = None
    first_audio: float | None = None
    eot_at: float | None = None
    eot_ms: float = 0.0
    last_word_wall: float | None = None
    speculative: bool = False
    latency_sent: bool = False


class FrameAssembler:
    """Asterisk drops partial frames, so audio leaves in whole 160-byte frames; the tail is padded with silence."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def push(self, data: bytes) -> bytes:
        self.buffer += data
        whole = len(self.buffer) - len(self.buffer) % FRAME
        out = bytes(self.buffer[:whole])
        del self.buffer[:whole]
        return out

    def finish(self) -> bytes:
        if not self.buffer:
            return b""
        out = bytes(self.buffer) + SILENCE * (FRAME - len(self.buffer))
        self.buffer.clear()
        return out


class AgentSession:
    def __init__(
        self,
        persona: Persona,
        listener,
        speaker,
        brain: Brain,
        out: asyncio.Queue,
        greeting_audio: bytes | None = None,
        settings: SessionSettings | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.persona = persona
        self.system = build_system_prompt(persona)
        self.listener = listener
        self.speaker = speaker
        self.brain = brain
        self.out = out
        self.greeting_audio = greeting_audio
        self.settings = settings or SessionSettings()
        self.clock = clock
        self.history: list[dict[str, Any]] = []
        self.utterances: dict[str, Utterance] = {}
        self.current: Utterance | None = None  # The utterance whose audio is going out now.
        self.assembler = FrameAssembler()
        self.response: Response | None = None
        self.held: Response | None = None
        self.pending_end: tuple[str, str, str] | None = None
        self.ended = asyncio.Event()
        self.tool_waiters: dict[str, asyncio.Future] = {}
        self.silence_task: asyncio.Task | None = None
        self.nudges = 0
        self.turns = 0
        self.started = clock()
        self.caller_talking = False
        self.barge_pending = False  # The caller may be talking over the agent; not yet sure it isn't echo.
        self.tasks: list[asyncio.Task] = []
        self.awaiting_audio: dict[str, Response] = {}  # First utterance of a turn -> its turn, for latency.

    # ----- inbound from the gateway -----

    async def on_audio(self, ulaw: bytes) -> None:
        await self.listener.send(ulaw)

    def on_tool_result(self, call_id: str, ok: bool, content: str) -> None:
        waiter = self.tool_waiters.pop(call_id, None)
        if waiter is not None and not waiter.done():
            waiter.set_result((ok, content))

    def on_mark_played(self, mark: str) -> None:
        key, _, position = mark.rpartition("|")
        utterance = self.utterances.get(key)
        if utterance is None:
            return
        if position == "end":
            utterance.played = utterance.sent
            utterance.played_all = True
            utterance.finished_at = self.clock()
            self._emit(transcript=agent_pb2.AgentTranscript(role="agent", text=utterance.text, interrupted=False))
            if utterance is self.current:
                self.current = None
                self._after_speaking()
        else:
            utterance.played = max(utterance.played, int(position))

    # ----- outbound -----

    def _emit(self, **payload) -> None:
        self.out.put_nowait(agent_pb2.AgentDown(**payload))

    def _send_audio(self, utterance: Utterance, data: bytes) -> None:
        # Asterisk closes the socket on a message over 65,500 bytes, so big buffers go in pieces.
        for start in range(0, len(data), 16000):
            piece = data[start : start + 16000]
            before = utterance.sent
            utterance.sent += len(piece)
            self._emit(audio=agent_pb2.AgentAudio(ulaw=piece))
            if utterance.sent // MARK_EVERY != before // MARK_EVERY:
                self._emit(mark=agent_pb2.AgentMark(id=f"{utterance.key}|{utterance.sent}"))

    def _end(self, action: str, department: str = "", reason: str = "") -> None:
        if self.ended.is_set():
            return
        self._emit(end=agent_pb2.AgentEnd(action=action, department_id=department, reason=reason))
        self.ended.set()

    # ----- the call -----

    async def run(self) -> None:
        try:
            await asyncio.gather(self.listener.connect(), self.speaker.connect())
        except Exception as err:
            log.warning("agent could not start: %s", type(err).__name__)
            self._end("fallback", reason=f"start_failed:{type(err).__name__}")
            return
        self.tasks = [asyncio.create_task(self._pump_speech()), asyncio.create_task(self._listen())]
        await self._greet()
        ended = asyncio.create_task(self.ended.wait())
        done, _ = await asyncio.wait([ended, *self.tasks], return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task is not ended and task.exception() is not None:
                log.warning("agent stopped: %s", type(task.exception()).__name__)
                self._end("fallback" if self.turns == 0 else "error", reason=type(task.exception()).__name__)
        ended.cancel()

    async def close(self) -> None:
        for task in [*self.tasks, self.silence_task, self.response and self.response.task, self.held and self.held.task]:
            if task is not None:
                task.cancel()
        for closer in (self.listener.close, self.speaker.close):
            with contextlib.suppress(Exception):
                await closer()

    async def _greet(self) -> None:
        if self.persona.purpose:
            # An outbound call opens with the model, which knows why it's calling.
            self._start_response("[The call has connected. Introduce yourself and say why you're calling.]", held=False)
            return
        if self.greeting_audio:
            utterance = self._new_utterance("greet", self.persona.greeting)
            self.current = utterance
            self._send_audio(utterance, self.assembler.push(self.greeting_audio))
            self._send_audio(utterance, self.assembler.finish())
            utterance.received_all = True
            self._emit(mark=agent_pb2.AgentMark(id=f"{utterance.key}|end"))
            return
        await self._speak_fixed(self.persona.greeting)

    def _new_utterance(self, key: str, text: str = "") -> Utterance:
        utterance = Utterance(key, text)
        self.utterances[key] = utterance
        return utterance

    async def _begin_speaking(self) -> Utterance:
        number = await self.speaker.begin()
        utterance = self._new_utterance(f"u{number}")
        self.current = utterance
        self.assembler = FrameAssembler()
        return utterance

    async def _speak_fixed(self, text: str) -> Utterance:
        """A line that isn't from the model: the greeting without a cache, or a silence check."""
        utterance = await self._begin_speaking()
        utterance.text = text
        await self.speaker.say(text)
        await self.speaker.finish()
        return utterance

    async def _pump_speech(self) -> None:
        async for number, data in self.speaker.audio():
            utterance = self.current
            if utterance is None or utterance.key != f"u{number}":
                continue  # From an utterance that was interrupted.
            if data is None:
                self._send_audio(utterance, self.assembler.finish())
                utterance.received_all = True
                self._emit(mark=agent_pb2.AgentMark(id=f"{utterance.key}|end"))
                continue
            response = self.awaiting_audio.pop(utterance.key, None)
            if response is not None:
                response.first_audio = self.clock()
                self._report_latency(response)
            self._send_audio(utterance, self.assembler.push(data))

    async def _listen(self) -> None:
        async for event in self.listener.events():
            await self._on_turn(event)

    async def _on_turn(self, event: TurnEvent) -> None:
        if self.ended.is_set():
            return
        audible = self._agent_audible()
        if event.kind == "start":
            self.caller_talking = True
            self._cancel_silence()
            if audible and len(self._new_words(event.transcript)) < BARGE_IN_WORDS:
                self.barge_pending = True  # Wait for words that aren't the agent's own.
            else:
                await self._barge_in()
        elif event.kind == "update":
            if self.barge_pending and len(self._new_words(event.transcript)) >= BARGE_IN_WORDS:
                self.barge_pending = False
                await self._barge_in()
        elif event.kind == "eager_end" and (audible or self.barge_pending) and self._is_echo(event.transcript):
            return
        elif event.kind == "eager_end" and self.settings.speculate:
            self._cancel_held()
            if event.transcript and self.response is None:
                self.held = self._new_response(event.transcript, held=True)
                self.held.task = asyncio.create_task(self._respond(self.held))
        elif event.kind == "resumed":
            self._cancel_held()
        elif event.kind == "end":
            self.caller_talking = False
            if (audible or self.barge_pending) and self._is_echo(event.transcript):
                # The agent heard itself: carry on as if nothing was said.
                self.barge_pending = False
                self._cancel_held()
                log.info("ignored echo of the agent's own words")
                if self.current is None or self.current.played_all:
                    self._after_speaking()
                return
            if self.barge_pending:
                self.barge_pending = False
                await self._barge_in()  # A short real reply, like "no", spoken over the agent.
            if not event.transcript:
                self._cancel_held()
                self._after_speaking()
                return
            self.turns += 1
            self._emit(transcript=agent_pb2.AgentTranscript(role="caller", text=event.transcript))
            eot_ms = (event.audio_clock_s - event.last_word_end_s) * 1000 if event.last_word_end_s is not None else 0.0
            if self._over_limits():
                self._cancel_held()
                await self._hand_off("I'm going to put you through to someone who can help further.")
                return
            held, self.held = self.held, None
            if held is not None and held.caller_text == event.transcript and not held.task.done():
                held.speculative = True
                response = held
            else:
                if held is not None:
                    held.task.cancel()
                response = self._new_response(event.transcript, held=False)
                response.task = asyncio.create_task(self._respond(response))
            response.eot_at = self.clock()
            response.last_word_wall = getattr(event, "last_word_wall", None)
            if response.last_word_wall is not None:
                eot_ms = (response.eot_at - response.last_word_wall) * 1000
            response.eot_ms = max(0.0, eot_ms)
            self.response = response
            response.released.set()

    # ----- responding -----

    def _new_response(self, caller_text: str, held: bool) -> Response:
        base = list(self.history)
        if base and base[-1]["role"] == "user" and isinstance(base[-1]["content"], str):
            # The caller spoke again before hearing anything back: one turn, not two.
            merged = base.pop()["content"].removesuffix("</caller>") + " " + caller_text + "</caller>"
            message = {"role": "user", "content": merged}
        else:
            message = caller_message(caller_text)
        response = Response(caller_text=caller_text, base=[*base, message], held=held, started=self.clock())
        if not held:
            response.released.set()
        return response

    def _start_response(self, caller_text: str, held: bool) -> None:
        response = self._new_response(caller_text, held)
        self.response = response
        response.task = asyncio.create_task(self._respond(response))

    async def _respond(self, response: Response) -> None:
        try:
            messages = response.messages = list(response.base)
            for _ in range(self.settings.max_rounds):
                reply = Reply()
                stream = self.brain.respond(self.system, messages, reply)
                queue: asyncio.Queue = asyncio.Queue()

                async def pull(stream=stream, queue=queue) -> None:
                    try:
                        async for text in stream:
                            if response.first_token is None:
                                response.first_token = self.clock()  # When the model produced it, held or not.
                            await queue.put(text)
                        await queue.put(None)
                    except Exception as err:
                        await queue.put(err)

                puller = asyncio.create_task(pull())
                try:
                    await response.released.wait()  # Held responses wait here, already thinking.
                    while (item := await queue.get()) is not None:
                        if isinstance(item, Exception):
                            raise item
                        if response.utterance is None:
                            response.utterance = await self._begin_speaking()
                            response.streaming = True
                            if not response.first_utterance:
                                response.first_utterance = response.utterance.key
                                response.first_text_sent = self.clock()
                                self.awaiting_audio[response.first_utterance] = response
                        response.utterance.text += item
                        await self.speaker.say(item)
                finally:
                    puller.cancel()
                if response.utterance is not None:
                    await self.speaker.finish()
                    response.utterance = None
                if not reply.content:
                    break
                messages.append({"role": "assistant", "content": reply.content})
                response.streaming = False
                if not reply.tool_uses:
                    break
                results, terminal = [], False
                for tool in reply.tool_uses:
                    ok, content = await self._run_tool(tool)
                    results.append((tool.id, ok, content))
                    terminal = terminal or (ok and tool.name in TERMINAL_TOOLS)
                messages.append(tool_result_message(results))
                if terminal:
                    break
            self._commit(response)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            log.warning("model failed: %s", type(err).__name__)
            if self.response is response:
                self.response = None
                await self._hand_off("Sorry, I'm having trouble on my side. Let me put you through to someone who can help.")
        finally:
            if self.response is response and response.task is asyncio.current_task():
                self.response = None
                if self.current is None or self.current.played_all:
                    self._after_speaking()  # Already played, or nothing to say: end, or wait for the caller.

    def _commit(self, response: Response, interrupted: Utterance | None = None) -> None:
        """Writes the turn into the history, keeping only what the caller heard."""
        messages = list(response.messages or response.base)
        if interrupted is not None:
            heard = interrupted.heard()
            if not response.streaming and len(messages) > len(response.base):
                # The round finished thinking, but its last text was cut off while playing.
                for message in reversed(messages):
                    if message["role"] == "assistant":
                        blocks = [b for b in message["content"] if b["type"] != "text"]
                        if heard:
                            blocks.insert(0, {"type": "text", "text": heard + " [interrupted]"})
                        message["content"] = blocks
                        break
            elif heard:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": heard + " [interrupted]"}]})
        # Drop empty assistant turns and any tool call left without its result.
        cleaned: list[dict[str, Any]] = []
        for message in messages:
            if message["role"] == "assistant":
                if not message["content"]:
                    continue
            cleaned.append(message)
        if cleaned and cleaned[-1]["role"] == "assistant":
            dangling = {b["id"] for b in cleaned[-1]["content"] if b["type"] == "tool_use"}
            if dangling:
                cleaned[-1]["content"] = [b for b in cleaned[-1]["content"] if b["type"] != "tool_use"]
                if not cleaned[-1]["content"]:
                    cleaned.pop()
        self.history = cleaned

    async def _run_tool(self, tool: ToolUse) -> tuple[bool, str]:
        call_id = tool.id
        waiter = asyncio.get_running_loop().create_future()
        self.tool_waiters[call_id] = waiter
        self._emit(tool_call=agent_pb2.ToolCall(call_id=call_id, name=tool.name, arguments_json=dumps(tool.input)))
        try:
            ok, content = await asyncio.wait_for(waiter, self.settings.tool_timeout_s)
        except TimeoutError:
            return False, dumps({"error": "the system didn't respond in time"})
        if ok and tool.name == "transfer_to_department":
            self.pending_end = ("transfer", str(tool.input.get("department", "")), str(tool.input.get("reason", ""))[:200])
        elif ok and tool.name == "end_call":
            self.pending_end = ("hangup", "", str(tool.input.get("reason", ""))[:200])
        return ok, content

    async def _hand_off(self, line: str) -> None:
        """Says one line and transfers to the tenant's fallback department (the gateway picks it)."""
        self.pending_end = ("transfer", "", "handoff")
        try:
            await self._speak_fixed(line)
        except Exception:
            self._end("transfer", "", "handoff")

    def _report_latency(self, response: Response) -> None:
        if response.latency_sent or response.eot_at is None:
            return
        response.latency_sent = True
        first_audio = response.first_audio or self.clock()
        self._emit(
            latency=agent_pb2.TurnLatency(
                end_of_turn_ms=response.eot_ms,
                llm_first_token_ms=((response.first_token or first_audio) - response.started) * 1000,
                tts_first_audio_ms=(first_audio - (response.first_text_sent or first_audio)) * 1000,
                voice_to_voice_ms=(first_audio - response.last_word_wall) * 1000
                if response.last_word_wall is not None
                else response.eot_ms + max(0.0, first_audio - response.eot_at) * 1000,
                speculative_hit=response.speculative,
            )
        )

    # ----- interruptions, silence and limits -----

    def _agent_audible(self) -> bool:
        now = self.clock()
        return any(
            (u.sent and not u.played_all and not u.interrupted) or (u.finished_at is not None and now - u.finished_at < ECHO_WINDOW_S)
            for u in self.utterances.values()
        )

    def _recent_agent_words(self) -> set[str]:
        now = self.clock()
        recent = [
            u.text for u in self.utterances.values()
            if not u.played_all or (u.finished_at is not None and now - u.finished_at < ECHO_WINDOW_S)
        ]  # fmt: skip
        return set(words_of(" ".join(recent)))

    def _new_words(self, transcript: str) -> list[str]:
        mine = self._recent_agent_words()
        return [w for w in words_of(transcript) if w not in mine]

    def _is_echo(self, transcript: str) -> bool:
        heard = words_of(transcript)
        if not heard:
            return True
        heard = [w for w in heard if w not in FILLER] or heard
        mine = self._recent_agent_words()
        return sum(w in mine for w in heard) / len(heard) >= ECHO_OVERLAP

    async def _barge_in(self) -> None:
        response = self.response
        speaking = self.current is not None and not self.current.played_all
        if response is None and not speaking:
            return
        interrupted = self.current if speaking else None
        if speaking:
            self._emit(flush=agent_pb2.AgentFlush())
            interrupted.interrupted = True
            interrupted.finished_at = self.clock()
            self._emit(transcript=agent_pb2.AgentTranscript(role="agent", text=interrupted.heard(), interrupted=True))
            self.current = None
            if interrupted.key.startswith("u"):
                await self.speaker.cancel()  # The cached greeting never used the speaker.
        if response is not None and response.task is not None and not response.task.done():
            response.task.cancel()
            self.response = None
            self._commit(response, interrupted)
        elif interrupted is not None and self.history:
            # The response had finished thinking but was still playing.
            last = Response(caller_text="", base=self.history[:-1], held=False)
            last.messages = list(self.history)
            self._commit(last, interrupted)
        if self.pending_end and self.pending_end[2] != "handoff":
            self.pending_end = None  # The caller wants to say something first.

    def _cancel_held(self) -> None:
        if self.held is not None:
            self.held.task.cancel()
            self.held = None

    def _after_speaking(self) -> None:
        if self.pending_end and self.response is None:
            self._end(*self.pending_end)
            return
        if self.response is None and not self.caller_talking:
            self._cancel_silence()
            self.silence_task = asyncio.create_task(self._silence_watch())

    def _cancel_silence(self) -> None:
        if self.silence_task is not None:
            self.silence_task.cancel()
            self.silence_task = None

    async def _silence_watch(self) -> None:
        await asyncio.sleep(self.settings.silence_s)
        self.nudges += 1
        if self.nudges == 1:
            line = "Are you still there?"
        else:
            line = "I'll let you go for now. Thanks for calling, goodbye."
            self.pending_end = ("hangup", "", "silence")
        self._note_agent_line(line)
        await self._speak_fixed(line)

    def _note_agent_line(self, line: str) -> None:
        if self.history and self.history[-1]["role"] == "assistant":
            self.history[-1]["content"] = [*self.history[-1]["content"], {"type": "text", "text": line}]

    def _over_limits(self) -> bool:
        return self.turns > self.settings.max_turns or self.clock() - self.started > self.settings.max_call_s

