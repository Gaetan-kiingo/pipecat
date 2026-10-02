"""P-29 (Swiss Voice Platform, experiment): the early start through the real user
aggregator and LLM service - what reaches the context, the model and the voice."""

import unittest
from unittest.mock import patch

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    FunctionCallFromLLM,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

FAST = {
    "SVP_EARLY_START": "on",
    "SVP_EARLY_START_AFTER_MS": "60",
    "SVP_EARLY_START_SETTLE_MS": "60",
}
WINDOW = 0.5
HISTORY = [{"role": "assistant", "content": "Bonjour, comment puis-je vous aider ?"}]


def interim(text):
    return InterimTranscriptionFrame(text=text, user_id="caller", timestamp="t")


def final(text):
    return TranscriptionFrame(text=text, user_id="caller", timestamp="t", finalized=True)


class AnsweringLLM(LLMService):
    """Answers every context frame with one sentence naming what it was asked."""

    def __init__(self, **kwargs):
        super().__init__(settings=LLMSettings(model="test-model"), **kwargs)
        self.requests: list[tuple[bool, list[dict]]] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        self.requests.append((frame.speculation, list(frame.context.messages)))
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(f"Answer to: {frame.context.messages[-1]['content']}"))
        await self.push_frame(LLMFullResponseEndFrame())


class ToolCallingLLM(LLMService):
    """Answers every context frame with a tool call."""

    def __init__(self, **kwargs):
        super().__init__(settings=LLMSettings(model="test-model"), **kwargs)
        self.requests: list[bool] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        self.requests.append(frame.speculation)
        await self.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="get_quote_status",
                    tool_call_id="call-1",
                    arguments={"number": "2026-02-158"},
                    context=frame.context,
                )
            ]
        )


class TestEarlyStartThroughThePipeline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        env = patch.dict("os.environ", FAST)
        env.start()
        self.addCleanup(env.stop)

    def _aggregator(self, context):
        return LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=WINDOW)]
                ),
            ),
        )

    @staticmethod
    def spoken(frames):
        return [f.text for f in frames if isinstance(f, LLMTextFrame)]

    async def test_a_confirmed_early_answer_is_one_request_one_message_and_spoken_after_the_turn(
        self,
    ):
        context = LLMContext(messages=list(HISTORY))
        llm = AnsweringLLM()
        down, _ = await run_test(
            Pipeline([self._aggregator(context), llm]),
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                interim("où êtes vous situé"),
                VADUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.25),  # the early request goes out here
                final("Où êtes-vous situé ?"),
                SleepFrame(sleep=WINDOW + 0.3),
            ],
        )
        # one request, speculative, against a copy: history + the hypothesis
        self.assertEqual([spec for spec, _ in llm.requests], [True])
        self.assertEqual(llm.requests[0][1][-1], {"role": "user", "content": "où êtes vous situé"})
        # the real conversation holds the committed transcript, once
        self.assertEqual(
            context.messages, [*HISTORY, {"role": "user", "content": "Où êtes-vous situé ?"}]
        )
        # the answer left the service only after the frame that ended the turn
        self.assertEqual(self.spoken(down), ["Answer to: où êtes vous situé"])
        ended = next(i for i, f in enumerate(down) if isinstance(f, UserStoppedSpeakingFrame))
        answered = next(i for i, f in enumerate(down) if isinstance(f, LLMFullResponseStartFrame))
        self.assertGreater(answered, ended)

    async def test_the_real_conversation_is_untouched_while_the_turn_is_open(self):
        context = LLMContext(messages=list(HISTORY))
        llm = AnsweringLLM()
        seen = []

        original = llm.process_frame

        async def watching(frame, direction):
            if isinstance(frame, LLMContextFrame):
                seen.append((frame.context is context, list(context.messages)))
            await original(frame, direction)

        llm.process_frame = watching
        await run_test(
            Pipeline([self._aggregator(context), llm]),
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                interim("bonjour"),
                VADUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.25),
                final("Bonjour."),
                SleepFrame(sleep=WINDOW + 0.3),
            ],
        )
        # the speculative request ran on another context object, and the shared
        # history had no caller message in it at that moment
        self.assertEqual(seen, [(False, HISTORY)])

    async def test_a_final_that_differs_is_answered_again_and_only_that_answer_is_spoken(self):
        context = LLMContext(messages=list(HISTORY))
        llm = AnsweringLLM()
        down, _ = await run_test(
            Pipeline([self._aggregator(context), llm]),
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                interim("je voudrais un devis"),
                VADUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.25),
                final("Je voudrais un devis existant."),
                SleepFrame(sleep=WINDOW + 0.3),
            ],
        )
        self.assertEqual([spec for spec, _ in llm.requests], [True, False])
        self.assertEqual(self.spoken(down), ["Answer to: Je voudrais un devis existant."])
        self.assertEqual(
            context.messages,
            [*HISTORY, {"role": "user", "content": "Je voudrais un devis existant."}],
        )

    async def test_a_caller_who_resumes_gets_one_answer_to_the_whole_sentence(self):
        context = LLMContext(messages=list(HISTORY))
        llm = AnsweringLLM()
        down, _ = await run_test(
            Pipeline([self._aggregator(context), llm]),
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                interim("non c'est zéro"),
                VADUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.25),  # an early request on half the sentence
                VADUserStartedSpeakingFrame(),
                interim("non c'est zéro deux"),
                VADUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.02),
                final("Non, c'est 02."),
                SleepFrame(sleep=WINDOW + 0.3),
            ],
        )
        # the half sentence was never spoken and never recorded
        self.assertEqual(self.spoken(down), ["Answer to: Non, c'est 02."])
        self.assertEqual(
            context.messages, [*HISTORY, {"role": "user", "content": "Non, c'est 02."}]
        )

    async def test_a_tool_is_never_called_on_an_early_answer(self):
        context = LLMContext(messages=list(HISTORY))
        llm = ToolCallingLLM()
        calls = []

        async def get_quote_status(params):
            calls.append(params.arguments)
            await params.result_callback({"found": True}, properties=None)

        llm.register_function("get_quote_status", get_quote_status)
        down, up = await run_test(
            Pipeline([self._aggregator(context), llm]),
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                interim("c'est le 2026-02-158"),
                VADUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.25),
                # the early request wanted the tool: withdrawn, nothing called yet
                final("C'est le 2026-02-158."),
                SleepFrame(sleep=WINDOW + 0.4),
            ],
        )
        self.assertEqual(llm.requests[:2], [True, False])
        # called once, by the request made when the turn had ended
        self.assertEqual(calls, [{"number": "2026-02-158"}])
        self.assertTrue(any(isinstance(f, EagerEndOfTurnCancelFrame) for f in [*down, *up]))

    async def test_off_means_the_stock_turn(self):
        with patch.dict("os.environ", {"SVP_EARLY_START": "off"}):
            context = LLMContext(messages=list(HISTORY))
            llm = AnsweringLLM()
            down, _ = await run_test(
                Pipeline([self._aggregator(context), llm]),
                frames_to_send=[
                    VADUserStartedSpeakingFrame(),
                    interim("bonjour"),
                    VADUserStoppedSpeakingFrame(),
                    SleepFrame(sleep=0.25),
                    final("Bonjour."),
                    SleepFrame(sleep=WINDOW + 0.3),
                ],
            )
        self.assertEqual([spec for spec, _ in llm.requests], [False])
        self.assertEqual(self.spoken(down), ["Answer to: Bonjour."])
