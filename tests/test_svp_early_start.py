"""P-29 (Swiss Voice Platform, experiment): the early start's match rule and observe mode."""

import asyncio
import unittest
from unittest.mock import patch

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    STTMetadataFrame,
    SVPTimingMarkFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy, svp_early_start
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

WINDOW = 0.45  # longer than the observer's last look (0.3 s)


def interim(text):
    return InterimTranscriptionFrame(text=text, user_id="caller", timestamp="")


def final(text):
    return TranscriptionFrame(text=text, user_id="caller", timestamp="", finalized=True)


class TestMatchRule(unittest.TestCase):
    def test_formatting_alone_is_not_a_difference(self):
        self.assertTrue(
            svp_early_start.matches("bonjour où êtes vous situé", "Bonjour, où êtes-vous situé ?")
        )
        self.assertTrue(
            svp_early_start.matches("c'est un devis existant", "C’est un devis existant.")
        )

    def test_another_word_is_a_difference(self):
        self.assertFalse(
            svp_early_start.matches("je voudrais un devis", "Je voudrais un devis existant.")
        )
        self.assertFalse(svp_early_start.matches("", "Oui."))

    def test_a_text_with_a_digit_must_be_identical(self):
        self.assertTrue(
            svp_early_start.matches("2026 tiret 02 tiret 158", "2026 tiret 02 tiret 158.")
        )
        # punctuation that changes a number is never ignored
        self.assertFalse(svp_early_start.matches("1.5", "15"))
        self.assertFalse(svp_early_start.matches("-5", "5"))
        self.assertFalse(svp_early_start.matches("2026-02-158", "2026 02 158"))
        # words against digits: answered again
        self.assertFalse(svp_early_start.matches("deux mille vingt-six", "2026"))

    def test_the_switch_is_off_unless_named(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(svp_early_start.mode(), "off")
        with patch.dict("os.environ", {"SVP_EARLY_START": "1"}):
            self.assertEqual(svp_early_start.mode(), "off")
        with patch.dict("os.environ", {"SVP_EARLY_START": " Observe "}):
            self.assertEqual(svp_early_start.mode(), "observe")


class TestObserveMode(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def _strategy(self, switch):
        with patch.dict("os.environ", {"SVP_EARLY_START": switch} if switch else {}, clear=False):
            if not switch:
                import os

                os.environ.pop("SVP_EARLY_START", None)
            strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=WINDOW)
        await strategy.setup(self.task_manager)
        await strategy.process_frame(STTMetadataFrame(service_name="test", ttfs_p99_latency=0.0))
        self.marks, self.stops = [], []

        @strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            if isinstance(frame, SVPTimingMarkFrame):
                self.marks.append(frame)

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            self.stops.append(params)

        await strategy.handle_user_turn_started()
        return strategy

    async def test_unset_means_stock_behaviour(self):
        strategy = await self._strategy(None)
        self.assertIsNone(strategy._early_observer)
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("bonjour"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await strategy.process_frame(final("Bonjour."))
        await asyncio.sleep(WINDOW + 0.1)
        self.assertEqual(len(self.stops), 1)
        self.assertEqual(self.marks, [])

    async def test_a_turn_whose_interim_was_the_final_is_reported_as_a_match(self):
        strategy = await self._strategy("observe")
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("où êtes vous situé"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.36)  # every look is taken on the interim
        await strategy.process_frame(final("Où êtes-vous situé ?"))
        await asyncio.sleep(WINDOW)
        self.assertEqual(len(self.stops), 1)  # the turn ends exactly as it does without us
        [mark] = self.marks
        self.assertEqual(
            (mark.mark, mark.data["mode"], mark.data["outcome"]),
            ("early_start", "observe", "ended"),
        )
        self.assertFalse(mark.data["digits"])
        self.assertEqual([look["after"] for look in mark.data["looks"]], [0.0, 0.1, 0.2, 0.3])
        self.assertTrue(all(look["ready"] and look["match"] for look in mark.data["looks"]))
        # the first look was the earliest, so it has the longest lead over the turn's end
        leads = [look["lead"] for look in mark.data["looks"]]
        self.assertEqual(leads, sorted(leads, reverse=True))
        self.assertGreater(leads[2], 0.2)

    async def test_a_last_word_missing_from_the_interim_is_reported_as_no_match(self):
        strategy = await self._strategy("observe")
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("je voudrais un devis"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.15)
        await strategy.process_frame(
            interim("je voudrais un devis existant")
        )  # the hypothesis lagged
        await asyncio.sleep(0.2)
        await strategy.process_frame(final("Je voudrais un devis existant."))
        await asyncio.sleep(WINDOW)
        [mark] = self.marks
        self.assertEqual([look["match"] for look in mark.data["looks"]], [False, False, True, True])

    async def test_a_number_written_otherwise_is_no_match(self):
        strategy = await self._strategy("observe")
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("deux mille vingt-six tiret zéro deux"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.36)
        await strategy.process_frame(final("2026 tiret 02."))
        await asyncio.sleep(WINDOW)
        [mark] = self.marks
        self.assertTrue(mark.data["digits"])
        self.assertFalse(any(look["match"] for look in mark.data["looks"]))

    async def test_segments_already_final_are_part_of_the_text(self):
        strategy = await self._strategy("observe")
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(final("Non merci."))
        await strategy.process_frame(interim("au revoir"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.36)
        await strategy.process_frame(final("Au revoir."))
        await asyncio.sleep(WINDOW)
        [mark] = self.marks
        self.assertEqual(mark.data["segments"], 2)
        self.assertTrue(all(look["match"] for look in mark.data["looks"]))

    async def test_a_caller_who_resumes_is_reported_as_a_start_for_nothing(self):
        strategy = await self._strategy("observe")
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("non c'est zéro"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.25)
        await strategy.process_frame(VADUserStartedSpeakingFrame())  # « ... deux »
        self.assertEqual(self.stops, [])
        [mark] = self.marks
        self.assertEqual(mark.data["outcome"], "resumed")
        self.assertEqual(mark.data["ready_at"], [0.0, 0.1, 0.2])

    async def test_a_mark_never_carries_a_transcript(self):
        strategy = await self._strategy("observe")
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("mon numéro est le 079"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.36)
        await strategy.process_frame(final("Mon numéro est le 079."))
        await asyncio.sleep(WINDOW)
        self.assertNotIn("079", str([mark.data for mark in self.marks]))
        self.assertNotIn("numéro", str([mark.data for mark in self.marks]))
