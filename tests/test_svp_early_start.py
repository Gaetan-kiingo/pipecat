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

WINDOW = 0.65  # longer than the observer's last look (0.5 s)


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
        await asyncio.sleep(0.56)  # every look is taken on the interim
        await strategy.process_frame(final("Où êtes-vous situé ?"))
        await asyncio.sleep(WINDOW)
        self.assertEqual(len(self.stops), 1)  # the turn ends exactly as it does without us
        [mark] = self.marks
        self.assertEqual(
            (mark.mark, mark.data["mode"], mark.data["outcome"]),
            ("early_start", "observe", "ended"),
        )
        self.assertFalse(mark.data["digits"])
        self.assertEqual(
            [look["after"] for look in mark.data["looks"]], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        )
        # the recogniser's own timing: the hypothesis was settled before silence was
        # detected, the final came 0.56 s after it, the turn ended at the window's close
        self.assertTrue(mark.data["last_interim_match"])
        self.assertLess(mark.data["last_interim_after"], 0.05)
        self.assertAlmostEqual(mark.data["final_after"], 0.56, delta=0.08)
        self.assertGreaterEqual(mark.data["turn_end_after"], mark.data["final_after"])
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
        self.assertEqual(
            [look["match"] for look in mark.data["looks"]],
            [False, False, True, True, True, True],
        )
        # the last two looks came after the final: nothing was left to gain there
        self.assertEqual(
            [look["final_in"] for look in mark.data["looks"]],
            [False, False, False, False, True, True],
        )
        self.assertTrue(mark.data["last_interim_match"])
        self.assertAlmostEqual(mark.data["last_interim_after"], 0.15, delta=0.06)

    async def test_a_number_written_otherwise_is_no_match(self):
        strategy = await self._strategy("observe")
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("deux mille vingt-six tiret zéro deux"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.56)
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
        await asyncio.sleep(0.56)
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
        await asyncio.sleep(0.56)
        await strategy.process_frame(final("Mon numéro est le 079."))
        await asyncio.sleep(WINDOW)
        self.assertNotIn("079", str([mark.data for mark in self.marks]))
        self.assertNotIn("numéro", str([mark.data for mark in self.marks]))


FAST = {
    "SVP_EARLY_START": "on",
    "SVP_EARLY_START_AFTER_MS": "60",
    "SVP_EARLY_START_SETTLE_MS": "60",
}
ON_WINDOW = 0.5


class TestEarlyStart(unittest.IsolatedAsyncioTestCase):
    """``on``: the strategy starts a speculative inference and resolves it at the turn's end."""

    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        self._env = patch.dict("os.environ", FAST)
        self._env.start()
        self.addCleanup(self._env.stop)

    async def _strategy(self):
        strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=ON_WINDOW)
        await strategy.setup(self.task_manager)
        await strategy.process_frame(STTMetadataFrame(service_name="test", ttfs_p99_latency=0.0))
        self.events, self.marks = [], []

        @strategy.event_handler("on_user_turn_inference_triggered")
        async def on_inference(strategy, speculation=None):
            self.events.append(("inference", speculation.text if speculation else None))

        @strategy.event_handler("on_user_turn_stopped")
        async def on_stopped(strategy, params):
            self.events.append(("stopped", params.confirms_speculation))
            await strategy.handle_user_turn_stopped()  # as the controller does

        @strategy.event_handler("on_user_turn_speculation_cancelled")
        async def on_cancelled(strategy):
            self.events.append(("cancelled", None))

        @strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            if isinstance(frame, SVPTimingMarkFrame):
                self.marks.append(frame.data)

        await strategy.handle_user_turn_started()
        return strategy

    async def test_a_settled_hypothesis_is_answered_early_and_confirmed_at_the_turns_end(self):
        strategy = await self._strategy()
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("où êtes vous situé"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.2)
        # started inside the wait, on the hypothesis; the turn is still open
        self.assertEqual(self.events, [("inference", "où êtes vous situé")])
        await strategy.process_frame(final("Où êtes-vous situé ?"))
        await asyncio.sleep(ON_WINDOW)
        # the turn ends once, confirming: no second inference
        self.assertEqual(self.events, [("inference", "où êtes vous situé"), ("stopped", True)])
        [mark] = self.marks
        self.assertEqual(
            (mark["mode"], mark["outcome"], mark["digits"]), ("on", "confirmed", False)
        )
        self.assertGreater(mark["lead"], 0.2)

    async def test_a_final_that_differs_withdraws_the_answer_and_the_turn_is_answered_again(self):
        strategy = await self._strategy()
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("je voudrais un devis"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.2)
        await strategy.process_frame(final("Je voudrais un devis existant."))
        await asyncio.sleep(ON_WINDOW)
        self.assertEqual(
            self.events,
            [
                ("inference", "je voudrais un devis"),
                ("cancelled", None),
                ("inference", None),  # the ordinary one, on the committed text
                ("stopped", False),
            ],
        )
        self.assertEqual([m["outcome"] for m in self.marks], ["redone"])

    async def test_a_number_written_otherwise_is_answered_again(self):
        strategy = await self._strategy()
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("1.5"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.2)
        await strategy.process_frame(final("15"))
        await asyncio.sleep(ON_WINDOW)
        self.assertEqual(
            [e[0] for e in self.events], ["inference", "cancelled", "inference", "stopped"]
        )
        self.assertTrue(self.marks[0]["digits"])

    async def test_a_caller_who_resumes_withdraws_the_answer_and_the_turn_stays_open(self):
        strategy = await self._strategy()
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("non c'est zéro"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.2)
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(self.events, [("inference", "non c'est zéro"), ("cancelled", None)])
        self.assertEqual([m["outcome"] for m in self.marks], ["resumed"])
        # the rest of the sentence: one ordinary... or early answer, never two
        await strategy.process_frame(interim("non c'est zéro deux"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.2)
        await strategy.process_frame(final("Non, c'est zéro deux."))
        await asyncio.sleep(ON_WINDOW)
        self.assertEqual(
            self.events[-2:], [("inference", "non c'est zéro deux"), ("stopped", True)]
        )

    async def test_a_hypothesis_that_still_changes_is_not_answered_until_it_settles(self):
        strategy = await self._strategy()
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("merci"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        for text in ("merci beaucoup", "merci beaucoup au", "merci beaucoup au revoir"):
            await asyncio.sleep(0.03)  # the recogniser is catching up with the audio
            await strategy.process_frame(interim(text))
        self.assertEqual(self.events, [])
        await asyncio.sleep(0.15)
        self.assertEqual(self.events, [("inference", "merci beaucoup au revoir")])

    async def test_a_hypothesis_that_changes_after_the_start_withdraws_it_and_starts_again(self):
        strategy = await self._strategy()
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("je voudrais un rendez-vous"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.15)
        await strategy.process_frame(interim("je voudrais un rendez-vous demain"))
        await asyncio.sleep(0.15)
        self.assertEqual(
            self.events,
            [
                ("inference", "je voudrais un rendez-vous"),
                ("cancelled", None),
                ("inference", "je voudrais un rendez-vous demain"),
            ],
        )
        self.assertEqual([m["outcome"] for m in self.marks], ["changed"])

    async def test_an_answer_the_llm_withdrew_leaves_the_turn_to_end_the_ordinary_way(self):
        from pipecat.frames.frames import EagerEndOfTurnCancelFrame

        strategy = await self._strategy()
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("quel est le statut de mon devis"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.2)
        await strategy.process_frame(EagerEndOfTurnCancelFrame())  # it wanted a tool call
        await strategy.process_frame(final("Quel est le statut de mon devis ?"))
        await asyncio.sleep(ON_WINDOW)
        self.assertEqual(
            self.events,
            [
                ("inference", "quel est le statut de mon devis"),
                ("inference", None),
                ("stopped", False),
            ],
        )
        self.assertEqual([m["outcome"] for m in self.marks], ["tool"])

    async def test_a_final_already_in_is_answered_inside_the_rest_of_the_window(self):
        strategy = await self._strategy()
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(final("Oui."))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.15)
        self.assertEqual(self.events, [("inference", "Oui.")])
        await asyncio.sleep(ON_WINDOW)
        self.assertEqual(self.events, [("inference", "Oui."), ("stopped", True)])

    async def test_a_mark_never_carries_a_transcript(self):
        strategy = await self._strategy()
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(interim("mon numéro est le 079"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.2)
        await strategy.process_frame(final("Mon numéro est le 079."))
        await asyncio.sleep(ON_WINDOW)
        self.assertNotIn("079", str(self.marks))
