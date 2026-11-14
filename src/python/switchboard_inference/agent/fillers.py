"""Short spoken acknowledgements that cover the wait while the answer is being worked out.

People don't sit in silence on the phone: they say "okay" or "let me check" and then answer. The agent does the
same, but only when the answer is genuinely slow (it waits a moment first, and plays nothing if speech has
already started), never after a goodbye, and never the same phrase twice in a row. The phrases are synthesised
once per voice at startup, so they play with no delay.
"""

from __future__ import annotations

import re

QUESTION = ("Let me check.", "One sec.", "Sure, let me see.", "Let me have a look.")
STATEMENT = ("Okay.", "Right, okay.", "Got it.", "Okay, one sec.")
NEUTRAL = ("Okay.", "One sec.", "Right.", "Let me see.")
ALL = tuple(dict.fromkeys(QUESTION + STATEMENT + NEUTRAL))

_QUESTION_START = re.compile(r"^(what|when|where|who|why|how|which|can|could|do|does|did|is|are|will|would|should|may|have|has)\b", re.I)
_CLOSING = re.compile(r"\b(thanks|thank you|bye|goodbye|cheers|that's all|that is all|no that's it)\b", re.I)


class FillerPicker:
    def __init__(self) -> None:
        self.last = ""
        self.count = 0

    def pick(self, caller_text: str | None) -> str | None:
        """A phrase for what the caller said, or None when an acknowledgement would sound wrong."""
        if caller_text is None:
            options = NEUTRAL  # Not transcribed yet: something that fits anything.
        else:
            text = caller_text.strip()
            words = text.split()
            if len(words) < 3 or _CLOSING.search(text):
                return None  # "Yes", "no", "thanks, bye": the answer is quick, and a filler would sound odd.
            is_question = text.endswith("?") or bool(_QUESTION_START.match(text))
            options = QUESTION if is_question else STATEMENT
        choices = [o for o in options if o != self.last] or list(options)
        choice = choices[self.count % len(choices)]
        self.count += 1
        self.last = choice
        return choice
