"""The agent's reasoning on OpenAI, as an alternative to Claude. Same prompt, same tools, same history.

The session keeps its history in one shape (Anthropic-style content blocks). This converts it to OpenAI chat
messages on the way out, and turns the streamed reply back into blocks, so either brain can take any turn.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from switchboard_inference.agent.brain import MAX_TOKENS, SUMMARY_PROMPT, TOOLS, Reply, ToolUse, _clean

DEFAULT_MODEL = "gpt-5.4-mini"

OPENAI_TOOLS = [
    {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOLS
]


def to_openai(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue
        if message["role"] == "assistant":
            text = "".join(b["text"] for b in content if b["type"] == "text")
            calls = [
                {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                for b in content
                if b["type"] == "tool_use"
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue
        for block in content:
            if block["type"] == "tool_result":
                out.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": block["content"]})
            elif block["type"] == "text":
                out.append({"role": "user", "content": block["text"]})
    return out


def _reasoning(model: str) -> dict[str, Any]:
    # Reasoning models think before the first word, which costs time on a call: ask for as little as possible.
    # gpt-5.4 and later accept "none"; gpt-5 and its mini and nano go no lower than "minimal".
    if model.startswith("gpt-5.") or model.startswith("gpt-6"):
        return {"reasoning_effort": "none"}
    return {"reasoning_effort": "minimal"} if model.startswith(("gpt-5", "o")) else {}


class OpenAIBrain:
    def __init__(self, client: Any = None, model: str = DEFAULT_MODEL, api_key: str | None = None):
        if client is None:
            import openai

            # Retries would arrive too late to be useful on a live call.
            client = openai.AsyncOpenAI(api_key=api_key, max_retries=0)
        self.client = client
        self.model = model

    async def respond(self, system: str, messages: list[dict[str, Any]], reply: Reply) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=to_openai(system, messages),
            tools=OPENAI_TOOLS,
            stream=True,
            max_completion_tokens=MAX_TOKENS,
            **_reasoning(self.model),
        )
        text = []
        calls: dict[int, dict[str, str]] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                text.append(delta.content)
                yield delta.content
            for call in delta.tool_calls or []:
                slot = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                if call.id:
                    slot["id"] = call.id
                if call.function and call.function.name:
                    slot["name"] = call.function.name
                if call.function and call.function.arguments:
                    slot["arguments"] += call.function.arguments
            if choice.finish_reason:
                reply.stop_reason = choice.finish_reason
        if text:
            reply.content.append({"type": "text", "text": "".join(text)})
        for index in sorted(calls):
            slot = calls[index]
            try:
                arguments = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}  # The gateway refuses it with a reason the model can read.
            reply.content.append({"type": "tool_use", "id": slot["id"], "name": slot["name"], "input": arguments})
            reply.tool_uses.append(ToolUse(slot["id"], slot["name"], arguments))

    async def summarize(self, business: str, lines: Sequence[tuple[str, str]], ending: str) -> tuple[str, str]:
        transcript = "\n".join(f"{'Caller' if role == 'caller' else 'Receptionist'}: {_clean(text)}" for role, text in lines)
        response = await self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=300,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": f"Business: {_clean(business)}\nHow it ended: {ending}\n<call>\n{transcript.replace('</call>', '')}\n</call>"},
            ],
            **_reasoning(self.model),
        )
        text = (response.choices[0].message.content or "").strip()
        outcome, kept = "resolved", []
        for line in text.splitlines():
            if line.strip().upper().startswith("OUTCOME:"):
                outcome = line.split(":", 1)[1].strip().lower() or outcome
            else:
                kept.append(line)
        return " ".join(" ".join(kept).split()), outcome
