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
- ``on``: the answer starts on that text while the turn is still open - once the
  recogniser's hypothesis has stopped changing (or its final is in) - against a
  provisional copy of the conversation. Everything it produces is held in the LLM
  service's speculation gate until the turn ends normally AND the turn's whole text
  still matches; a tool call or a step change withdraws it instead of running. A caller
  who resumes, a hypothesis that changes, or a final that differs withdraw it, and the
  turn is then answered the ordinary way.
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
OFFSETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

_SPACES = re.compile(r"\s+")
_CLOSING = " .!?…"
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_APOSTROPHES = re.compile(r"['’ʼ]")


def _secs(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, "").strip()) / 1000)
    except ValueError:
        return default


def start_after() -> float:
    """Seconds after the voice-activity stop before an early start may happen."""
    return _secs("SVP_EARLY_START_AFTER_MS", 0.15)


def settle() -> float:
    """Seconds the hypothesis must have gone unchanged before an early start."""
    return _secs("SVP_EARLY_START_SETTLE_MS", 0.15)


# seconds after the voice-activity stop past which no early start is attempted
GIVE_UP_AFTER = 0.55


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
        # the recogniser's own timing: when silence was detected, when the hypothesis
        # last changed, what it was just before the final replaced it, when that came
        self._vad_stopped_at: float | None = None
        self._interim_at: float | None = None
        self._before_final: tuple[str, float | None, float] | None = None

    def on_interim(self, text: str) -> None:
        """Keep the hypothesis of the segment in flight."""
        if text != self._interim:
            self._interim_at = time.time()
        self._interim = text

    def on_final(self, text: str) -> None:
        """Keep a segment the recogniser committed."""
        self._before_final = (joined(self._finals, self._interim), self._interim_at, time.time())
        self._finals.append(text)
        self._interim = ""  # the hypothesis in flight was this segment's

    async def on_vad_stopped(self) -> None:
        """Start looking at what an early start would have."""
        await self._stop_looking()
        self._snapshots = []
        self._vad_stopped_at = time.time()
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
            data: dict[str, Any] = {
                "mode": "observe",
                "outcome": "ended",
                "digits": has_digit(final),
                "chars": len(final),
                "segments": len(self._finals),
                "looks": looks,
            }
            stopped = self._vad_stopped_at
            if self._before_final and stopped:
                last_text, last_at, final_at = self._before_final
                # the best an early start could do: the last hypothesis before the final
                data["last_interim_match"] = matches(last_text, final)
                data["last_interim_after"] = round(last_at - stopped, 3) if last_at else None
                data["final_after"] = round(final_at - stopped, 3)
                data["turn_end_after"] = round(now - stopped, 3)
            await self._emit(data)
        self._snapshots = []

    async def reset(self) -> None:
        """Forget the turn."""
        await self._stop_looking()
        self._finals = []
        self._interim = ""
        self._snapshots = []
        self._vad_stopped_at = None
        self._interim_at = None
        self._before_final = None

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


class EarlyStarter:
    """``on`` mode: start the answer on the settled hypothesis, confirm it at the turn's end.

    The owner (the turn-stop strategy) keeps deciding when the turn ends; this only
    starts a speculative inference inside the wait and says, when the turn ends,
    whether that inference still answers it. One speculation is in flight at most.
    """

    def __init__(
        self,
        *,
        speculate: Callable[[str], Awaitable[None]],
        withdraw: Callable[[], Awaitable[None]],
        emit: Callable[[dict[str, Any]], Awaitable[None]],
        is_speaking: Callable[[], bool],
        create_task: Callable[[Any, str], asyncio.Task],
        cancel_task: Callable[[asyncio.Task], Awaitable[None]],
    ):
        """Initialize the starter.

        Args:
            speculate: Starts a speculative inference on the given turn text.
            withdraw: Withdraws the speculative inference in flight.
            emit: Reports how an attempt ended (the mark's data).
            is_speaking: Whether voice activity says the caller is speaking.
            create_task: Starts a task the owner tracks.
            cancel_task: Cancels such a task.
        """
        self._speculate = speculate
        self._withdraw = withdraw
        self._emit = emit
        self._is_speaking = is_speaking
        self._create_task = create_task
        self._cancel_task = cancel_task
        self._finals: list[str] = []
        self._interim = ""
        self._interim_at: float | None = None
        self._final_in = False
        self._task: asyncio.Task | None = None
        self._vad_stopped_at: float | None = None
        # the text the inference in flight was started on, and when
        self._text: str | None = None
        self._started_at: float = 0.0

    @property
    def speculating(self) -> bool:
        """Whether a speculative inference is in flight."""
        return self._text is not None

    async def on_interim(self, text: str) -> None:
        """Keep the hypothesis; a change voids an answer started on the old one."""
        if text == self._interim:
            return
        self._interim = text
        self._interim_at = time.time()
        self._final_in = False
        if self._text is not None and joined(self._finals, text) != self._text:
            await self._end("changed", withdraw=True)
            await self._wait_again()

    async def on_final(self, text: str) -> None:
        """Keep a committed segment; a final that differs voids the answer now."""
        self._finals.append(text)
        self._interim = ""
        self._final_in = True
        if self._text is not None and not matches(self._text, joined(self._finals)):
            await self._end("redone", withdraw=True)

    async def on_vad_stopped(self) -> None:
        """The caller fell silent: wait for the hypothesis to settle, then start."""
        self._vad_stopped_at = time.time()
        await self._wait_again()

    async def on_vad_started(self) -> None:
        """The caller resumed: whatever was started answers half a sentence."""
        await self._stop_waiting()
        if self._text is not None:
            await self._end("resumed", withdraw=True)

    async def on_withdrawn_elsewhere(self) -> None:
        """The LLM service withdrew the answer itself (it wanted a tool call)."""
        if self._text is not None:
            await self._end("tool", withdraw=False)

    async def on_turn_stopping(self) -> bool:
        """The turn is ending: say whether the answer in flight still stands.

        Returns:
            True when the turn's whole text matches what the answer was started
            on, so the owner only finalizes. False when there is no answer in
            flight, or it was withdrawn here because the text differs.
        """
        await self._stop_waiting()
        if self._text is None:
            return False
        if matches(self._text, joined(self._finals)) and not self._interim:
            await self._end("confirmed", withdraw=False)
            return True
        await self._end("redone", withdraw=True)
        return False

    async def reset(self) -> None:
        """Forget the turn, withdrawing an answer nothing resolved."""
        await self._stop_waiting()
        if self._text is not None:
            await self._end("unresolved", withdraw=True)
        self._finals = []
        self._interim = ""
        self._interim_at = None
        self._final_in = False
        self._vad_stopped_at = None

    async def _wait_again(self) -> None:
        await self._stop_waiting()
        if self._vad_stopped_at is None or self._is_speaking():
            return
        self._task = self._create_task(self._wait_then_start(), "svp_early_start::wait")

    async def _wait_then_start(self) -> None:
        stopped = self._vad_stopped_at or time.time()
        after, settled = start_after(), settle()
        try:
            while True:
                now = time.time()
                if now - stopped >= GIVE_UP_AFTER:
                    return
                waited = now - stopped >= after
                quiet = self._final_in or (
                    self._interim_at is not None and now - self._interim_at >= settled
                )
                if waited and quiet:
                    break
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            return
        finally:
            self._task = None

        text = joined(self._finals, self._interim)
        if not text or self._is_speaking() or self._text is not None:
            return
        self._text = text
        self._started_at = time.time()
        logger.debug(f"svp early start: answering [{text}] before the turn ends")
        await self._speculate(text)

    async def _end(self, outcome: str, *, withdraw: bool) -> None:
        text, self._text = self._text, None
        if text is None:
            return
        now = time.time()
        if withdraw:
            await self._withdraw()
        await self._emit(
            {
                "mode": "on",
                "outcome": outcome,
                "digits": has_digit(text),
                "after": round(self._started_at - (self._vad_stopped_at or self._started_at), 3),
                "lead": round(now - self._started_at, 3),
            }
        )

    async def _stop_waiting(self) -> None:
        if self._task:
            task, self._task = self._task, None
            await self._cancel_task(task)
