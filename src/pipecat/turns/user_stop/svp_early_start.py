#
# Swiss Voice Platform (ADR-002, P-29 - an experiment on its own branch).
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Starting the answer on an interim transcript: the switch, the match rule, the observer.

Azure's final transcript arrives 0.7 to 1.1 s after the caller stops; its interim
hypothesis is there earlier. ``SVP_EARLY_START`` says what the turn-stop strategy does
with it:

- unset or ``off``: stock behaviour.
- ``observe``: nothing changes for the caller. At fixed moments after the voice
  activity stop the strategy notes the text it would have started on (the segments
  already final plus the interim), and when the turn ends it reports, as a timing mark,
  whether that text was there, whether it matched what the turn finally held, and how
  much earlier it was. A caller who resumes speaking is reported too: a start that
  would have been thrown away.

The match rule is conservative: a text with a digit must be identical apart from letter
case, spacing and a closing sentence mark - « 1.5 » is not « 15 » - and only a text
without digits has its punctuation ignored. A needless second answer is cheaper than a
changed number.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

MARK = "early_start"
# seconds after the voice-activity stop at which the observer looks
OFFSETS = (0.0, 0.1, 0.2, 0.3)

_SPACES = re.compile(r"\s+")
_CLOSING = " .!?…"
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_APOSTROPHES = re.compile(r"['’ʼ]")


def mode() -> str:
    """The switch: ``off`` (stock), ``observe`` or ``on``."""
    value = os.getenv("SVP_EARLY_START", "").strip().lower()
    return value if value in ("observe", "on") else "off"


def has_digit(text: str) -> bool:
    """Whether the text holds a digit, which makes the match rule strict."""
    return any(char.isdigit() for char in text)


def _basic(text: str) -> str:
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def matches(provisional: str, final: str) -> bool:
    """Whether an answer started on ``provisional`` still answers ``final``."""
    a, b = _basic(provisional), _basic(final)
    if not a or not b:
        return False
    if has_digit(a) or has_digit(b):
        return a.rstrip(_CLOSING) == b.rstrip(_CLOSING)

    def words(text: str) -> str:
        text = _APOSTROPHES.sub("", text)
        return _SPACES.sub(" ", _PUNCTUATION.sub(" ", text)).strip()

    return words(a) == words(b)


def joined(parts: list[str], interim: str = "") -> str:
    """The turn's text: its final segments, then the interim hypothesis if any."""
    return " ".join(part.strip() for part in [*parts, interim] if part and part.strip())


class EarlyStartObserver:
    """``observe`` mode: what an early start would have had, reported when the turn ends.

    The marks carry times, lengths and booleans - never a transcript.
    """

    def __init__(
        self,
        emit: Callable[[dict[str, Any]], Awaitable[None]],
        create_task: Callable[[Any, str], asyncio.Task],
        cancel_task: Callable[[asyncio.Task], Awaitable[None]],
    ):
        """Initialize the observer.

        Args:
            emit: Reports one observation (the mark's data).
            create_task: Starts a task the owner tracks.
            cancel_task: Cancels such a task.
        """
        self._emit = emit
        self._create_task = create_task
        self._cancel_task = cancel_task
        self._finals: list[str] = []
        self._interim = ""
        self._snapshots: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None

    def on_interim(self, text: str) -> None:
        """Keep the hypothesis of the segment in flight."""
        self._interim = text

    def on_final(self, text: str) -> None:
        """Keep a segment the recogniser committed."""
        self._finals.append(text)
        self._interim = ""  # the hypothesis in flight was this segment's

    async def on_vad_stopped(self) -> None:
        """Start looking at what an early start would have."""
        await self._stop_looking()
        self._snapshots = []
        self._task = self._create_task(self._look(), "svp_early_start::look")

    async def on_vad_started(self) -> None:
        """The caller resumed: a start made at any of these moments was for nothing."""
        await self._stop_looking()
        if self._snapshots:
            await self._emit(
                {
                    "mode": "observe",
                    "outcome": "resumed",
                    "ready_at": [s["after"] for s in self._snapshots if s["text"]],
                    "after_secs": round(time.time() - self._snapshots[0]["at"], 3),
                }
            )
        self._snapshots = []

    async def on_turn_stopped(self) -> None:
        """Report each look against the text the turn finally held."""
        await self._stop_looking()
        final = joined(self._finals)
        now = time.time()
        if self._snapshots and final:
            looks = []
            for snapshot in self._snapshots:
                same = matches(snapshot["text"], final)
                looks.append(
                    {
                        "after": snapshot["after"],
                        "ready": bool(snapshot["text"]),
                        "match": same,
                        "final_in": snapshot["final_in"],
                        "lead": round(now - snapshot["at"], 3),
                    }
                )
                if snapshot["text"] and not same:
                    logger.debug(
                        f"svp early start (observe): at +{snapshot['after']}s "
                        f"[{snapshot['text']}] != final [{final}]"
                    )
            await self._emit(
                {
                    "mode": "observe",
                    "outcome": "ended",
                    "digits": has_digit(final),
                    "chars": len(final),
                    "segments": len(self._finals),
                    "looks": looks,
                }
            )
        self._snapshots = []

    async def reset(self) -> None:
        """Forget the turn."""
        await self._stop_looking()
        self._finals = []
        self._interim = ""
        self._snapshots = []

    async def _look(self) -> None:
        started = time.time()
        try:
            for after in OFFSETS:
                wait = started + after - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._snapshots.append(
                    {
                        "after": after,
                        "at": time.time(),
                        "text": joined(self._finals, self._interim),
                        # the segment in flight was already final: nothing left to gain
                        "final_in": bool(self._finals) and not self._interim,
                    }
                )
        except asyncio.CancelledError:
            return
        finally:
            self._task = None

    async def _stop_looking(self) -> None:
        if self._task:
            task, self._task = self._task, None
            await self._cancel_task(task)
