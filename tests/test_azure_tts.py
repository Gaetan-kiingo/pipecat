#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AzureTTSService cross-thread audio delivery.

Azure's Speech SDK fires its synthesis callbacks from native (non-event-loop)
threads. Those callbacks must deliver to the awaiting ``run_tts`` getter even when
the event loop is otherwise idle — e.g. a headless pipeline with no output transport
pumping audio. A bare ``asyncio.Queue.put_nowait()`` from another thread does NOT
wake an idle selector, so the audio sits unread; the callbacks marshal onto the loop
with ``asyncio.run_coroutine_threadsafe(queue.put(...), self.get_event_loop())``.

These are deterministic regression tests: the getter is parked on an idle loop
*first*, then the callback is fired from a real thread. With the fix the getter wakes
in ~ms; without it the loop stays blocked in ``select()`` until the ``wait_for`` timer
(seconds), so the elapsed assertion fails (or ``wait_for`` raises ``TimeoutError``).
"""

import asyncio
import threading
import time
from unittest.mock import Mock

import pytest

pytest.importorskip("azure.cognitiveservices.speech")

from azure.cognitiveservices.speech import CancellationReason  # noqa: E402

from pipecat.services.azure.tts import AzureHttpTTSService, AzureTTSService  # noqa: E402

SSML_SERVICE_CLASSES = (AzureTTSService, AzureHttpTTSService)

# The thread fires after this delay, by which point the awaiting getter has parked
# the loop in select() — so only the callback itself can wake it.
_FIRE_DELAY = 0.1
# Generous wait so the buggy path blocks on the timer; tight bound so the fix (~ms
# after _FIRE_DELAY) passes while the bug (~_WAIT) fails.
_WAIT = 5.0
_MAX_DELIVERY = 1.0


def _make_service() -> AzureTTSService:
    svc = AzureTTSService(api_key="test-key", region="eastus")
    # The SDK callbacks call get_event_loop(); without a started pipeline there is no
    # task manager, so point it at the running test loop.
    loop = asyncio.get_running_loop()
    svc.get_event_loop = lambda: loop
    return svc


async def _assert_idle_loop_wakeup(get_coro, fire):
    """Park ``get_coro`` on an idle loop, then ``fire()`` the callback from a thread.

    Returns the value the getter received; asserts it arrived promptly (i.e. the
    cross-thread put woke the idle loop rather than waiting for the ``wait_for`` timer).
    """
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(get_coro)
    t0 = loop.time()
    threading.Thread(target=lambda: (time.sleep(_FIRE_DELAY), fire())).start()
    result = await asyncio.wait_for(task, timeout=_WAIT)
    assert loop.time() - t0 < _MAX_DELIVERY
    return result


@pytest.mark.asyncio
async def test_synthesizing_audio_wakes_idle_loop():
    """Audio pushed from an SDK thread reaches a parked getter on an idle loop."""
    svc = _make_service()

    audio = b"\x00\x01" * 256
    evt = Mock()
    evt.result.audio_data = audio

    data = await _assert_idle_loop_wakeup(
        svc._audio_queue.get(), lambda: svc._handle_synthesizing(evt)
    )
    assert data == audio


@pytest.mark.asyncio
async def test_canceled_error_wakes_idle_loop():
    """A non-user cancellation delivers its error to a parked getter on an idle loop."""
    svc = _make_service()

    evt = Mock()
    evt.result.cancellation_details.reason = CancellationReason.Error
    evt.result.cancellation_details.error_details = "boom"

    item = await _assert_idle_loop_wakeup(svc._audio_queue.get(), lambda: svc._handle_canceled(evt))
    assert isinstance(item, Exception)
    assert "boom" in str(item)


@pytest.mark.asyncio
async def test_completion_sentinel_wakes_idle_loop():
    """The completion sentinel reaches the word-boundary getter on an idle loop.

    ``_handle_completed`` routes completion through the word-boundary queue, whose
    getter is the (loop-side) word-processor task; that handoff is also a cross-thread
    put and must wake an idle loop.
    """
    svc = _make_service()

    evt = Mock()
    evt.result.audio_duration = None  # skip duration bookkeeping

    item = await _assert_idle_loop_wakeup(
        svc._word_boundary_queue.get(), lambda: svc._handle_completed(evt)
    )
    assert item is None


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_construct_ssml_default_has_no_lang_element(service_class):
    """Default settings emit no <lang> element anywhere in the output."""
    service = service_class(api_key="test-key", region="eastus")

    ssml = service._construct_ssml("Hello there.")

    assert "<lang" not in ssml
    assert ssml == (
        "<speak version='1.0' xml:lang='en-US' "
        "xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='http://www.w3.org/2001/mstts'>"
        "<voice name='en-US-SaraNeural'>"
        "<mstts:silence type='Sentenceboundary' value='20ms' />"
        "Hello there."
        "</voice></speak>"
    )


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_construct_ssml_force_locale_false_matches_default(service_class):
    """force_locale=False explicitly must match the unset default byte-for-byte."""
    default_service = service_class(api_key="test-key", region="eastus")
    explicit_false_service = service_class(
        api_key="test-key",
        region="eastus",
        settings=service_class.Settings(force_locale=False),
    )

    assert default_service._construct_ssml(
        "Hello there."
    ) == explicit_false_service._construct_ssml("Hello there.")


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_construct_ssml_force_locale_wraps_text_in_lang_element(service_class):
    """force_locale=True wraps the text in <lang xml:lang> inside <voice>."""
    service = service_class(
        api_key="test-key",
        region="eastus",
        settings=service_class.Settings(language="en-GB", force_locale=True),
    )

    ssml = service._construct_ssml("Hello there.")

    assert ssml == (
        "<speak version='1.0' xml:lang='en-GB' "
        "xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='http://www.w3.org/2001/mstts'>"
        "<voice name='en-US-SaraNeural'>"
        "<mstts:silence type='Sentenceboundary' value='20ms' />"
        "<lang xml:lang='en-GB'>"
        "Hello there."
        "</lang>"
        "</voice></speak>"
    )


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_construct_ssml_force_locale_wraps_nested_style_and_prosody(service_class):
    """force_locale=True wraps the entire nested style/prosody/emphasis block,
    not just the raw text.
    """
    service = service_class(
        api_key="test-key",
        region="eastus",
        settings=service_class.Settings(
            language="en-GB",
            force_locale=True,
            style="cheerful",
            rate="slow",
            emphasis="strong",
        ),
    )

    ssml = service._construct_ssml("Hello there.")

    assert ssml == (
        "<speak version='1.0' xml:lang='en-GB' "
        "xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='http://www.w3.org/2001/mstts'>"
        "<voice name='en-US-SaraNeural'>"
        "<mstts:silence type='Sentenceboundary' value='20ms' />"
        "<lang xml:lang='en-GB'>"
        "<mstts:express-as style='cheerful'>"
        "<prosody rate='slow'>"
        "<emphasis level='strong'>"
        "Hello there."
        "</emphasis>"
        "</prosody>"
        "</mstts:express-as>"
        "</lang>"
        "</voice></speak>"
    )


# --- P-30 (ADR-002): the socket's warm-up never runs on the event loop -----------------
#
# On 2026-10-05 Azure's blocking ``Connection.open`` was called on the event loop at the
# start of a caller's turn, 50 ms after a sentence had started; it never returned and
# every call of the process stopped. These tests use a connection whose ``open`` does
# not return until the test lets it.


class _HungConnection:
    """A synthesis connection whose ``open`` blocks until ``release`` is set."""

    def __init__(self):
        self.release = threading.Event()
        self.opens = 0

    def open(self, _for_continuous):
        self.opens += 1
        self.release.wait(timeout=30)


def _warm_service(connection) -> AzureTTSService:
    svc = _make_service()
    svc._synthesizer_connection = connection
    svc._keepalive_connection = True
    return svc


@pytest.mark.asyncio
async def test_p30_an_open_that_never_returns_does_not_stop_the_loop():
    from pipecat.frames.frames import UserStartedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.services.tts_service import TTSService

    connection = _HungConnection()
    svc = _warm_service(connection)
    seen = []

    async def _passed_on(self, frame, direction):
        seen.append(frame)

    original = TTSService.process_frame
    TTSService.process_frame = _passed_on
    loop = asyncio.get_running_loop()
    try:
        t0 = loop.time()
        # the caller starts speaking: the frame is processed at once although the
        # open it asks for does not return
        await asyncio.wait_for(
            svc.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM), timeout=1.0
        )
        assert loop.time() - t0 < 0.2
        assert len(seen) == 1  # and the frame went on down the pipeline
        # the loop keeps its time while the open hangs
        t1 = loop.time()
        await asyncio.sleep(0.05)
        assert loop.time() - t1 < 0.2
        assert connection.opens == 1 and svc._warm_thread.is_alive()
        assert svc._warm_thread.daemon and svc._warm_thread is not threading.main_thread()
    finally:
        TTSService.process_frame = original
        connection.release.set()


@pytest.mark.asyncio
async def test_p30_one_open_at_a_time_and_a_hung_one_is_reported_once(monkeypatch):
    import pipecat.services.azure.tts as azure_tts

    connection = _HungConnection()
    svc = _warm_service(connection)
    warnings = []
    handle = azure_tts.logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    try:
        first = svc._warm_connection("user-start")
        assert first is not None
        # while it has not returned, later caller turns start no other thread
        assert svc._warm_connection("user-start") is None
        assert svc._warm_connection("disconnect") is None
        assert connection.opens == 1 and not warnings
        # once it is older than the limit it is named in the log, once
        monkeypatch.setattr(azure_tts, "_WARM_STUCK_SECONDS", 0.0)
        assert svc._warm_connection("user-start") is None
        assert svc._warm_connection("user-start") is None
        assert len(warnings) == 1 and "has not returned" in warnings[0]
        # when it returns, the next turn warms the socket again
        connection.release.set()
        first.join(timeout=2)
        assert not first.is_alive()
        second = svc._warm_connection("user-start")
        assert second is not None and second is not first
        second.join(timeout=2)
        assert connection.opens == 2
    finally:
        azure_tts.logger.remove(handle)
        connection.release.set()


@pytest.mark.asyncio
async def test_p30_nothing_is_opened_while_a_sentence_is_being_synthesized():
    connection = _HungConnection()
    connection.release.set()  # an open would return at once: it must not even be asked
    svc = _warm_service(connection)
    svc._synthesis_under_way = True  # the situation of 2026-10-05: the agent speaks
    assert svc._warm_connection("user-start") is None
    assert connection.opens == 0
    # and no warm-up without a connection, or once the call is being torn down
    svc._synthesis_under_way = False
    svc._keepalive_connection = False
    assert svc._warm_connection("user-start") is None
    svc._keepalive_connection = True
    svc._synthesizer_connection = None
    assert svc._warm_connection("user-start") is None
    assert connection.opens == 0


@pytest.mark.asyncio
async def test_p30_the_start_waits_for_its_first_open_for_a_bounded_time():
    connection = _HungConnection()
    svc = _warm_service(connection)
    loop = asyncio.get_running_loop()
    try:
        # a hung open: the start goes on after the limit, and the loop ran meanwhile
        ticks = 0

        async def _tick():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        ticker = asyncio.ensure_future(_tick())
        t0 = loop.time()
        returned = await svc._wait_for_warm(svc._warm_connection("start"), 0.3)
        waited = loop.time() - t0
        ticker.cancel()
        assert returned is False and 0.3 <= waited < 1.0
        assert ticks >= 10  # other calls' work went on during the wait
    finally:
        connection.release.set()
    # an open that returns is waited for, and only as long as it takes
    quick = _HungConnection()
    quick.release.set()
    svc2 = _warm_service(quick)
    t0 = loop.time()
    assert await svc2._wait_for_warm(svc2._warm_connection("start"), 3.0) is True
    assert loop.time() - t0 < 0.5 and quick.opens == 1
    assert await svc2._wait_for_warm(None, 3.0) is True


@pytest.mark.asyncio
async def test_p30_a_sentence_marks_the_socket_in_use_until_it_ends_or_is_interrupted():
    svc = _make_service()
    states = []

    class _Synthesizer:
        def speak_ssml_async(self, ssml):
            states.append(svc._synthesis_under_way)
            svc._audio_queue.put_nowait(b"\x00\x00" * 80)
            svc._audio_queue.put_nowait(None)

    async def _no_metrics(text):
        return None

    svc._speech_synthesizer = _Synthesizer()
    svc.start_tts_usage_metrics = _no_metrics
    svc._construct_ssml = lambda text: "<speak/>"
    assert svc._synthesis_under_way is False
    frames = [frame async for frame in svc.run_tts("Bonjour.", "ctx-1")]
    assert states == [True] and len(frames) == 1  # in use while Azure synthesizes
    assert svc._synthesis_under_way is False  # free again when the sentence ends
    # abandoned after its first audio: the mark is cleared when the generator is
    # finalized (run_tts is wrapped by the tracing decorator, so closing the wrapper
    # finalizes it on the loop's next turns, not inside aclose)
    stream = svc.run_tts("Une phrase interrompue.", "ctx-2")
    await stream.__anext__()
    assert svc._synthesis_under_way is True
    await stream.aclose()
    del stream
    import gc

    for _ in range(20):
        gc.collect()
        await asyncio.sleep(0.01)
        if not svc._synthesis_under_way:
            break
    assert svc._synthesis_under_way is False


@pytest.mark.asyncio
async def test_p30_an_interruption_frees_the_socket_at_once():
    from pipecat.frames.frames import InterruptionFrame
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.services.tts_service import TTSService

    svc = _make_service()

    class _Stopped:
        def get(self):
            return None

    class _Synthesizer:
        def stop_speaking_async(self):
            return _Stopped()

    async def _nothing(*args, **kwargs):
        return None

    svc._speech_synthesizer = _Synthesizer()
    svc.stop_all_metrics = _nothing
    original = TTSService._handle_interruption
    TTSService._handle_interruption = _nothing
    try:
        svc._synthesis_under_way = True  # the agent was speaking
        await svc._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        assert svc._synthesis_under_way is False  # the next caller turn may warm it
    finally:
        TTSService._handle_interruption = original
