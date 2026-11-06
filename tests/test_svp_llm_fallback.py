"""P-46 (Swiss Voice Platform, ADR-065): the in-call fallback of an OpenAI-compatible engine.

The primary engine has ``after_secs`` to give its first chunk; a failure or a silence until
then sends the same request to the fallback in the same turn, with the fallback's own model
and extra parameters; after two such turns the rest of the call stays on the fallback."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import APIStatusError

from pipecat.services.openai.llm import OpenAILLMService


def words(text):
    """A chunk carrying words, as the OpenAI-compatible stream gives it."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))])


def empty():
    """A chunk with no words: the role-only first delta, or an empty content."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="", tool_calls=None))])


def tool_call():
    call = SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(name="book", arguments=""))
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[call]))])


def pause(seconds, chunk):
    """A chunk that comes only after `seconds`."""
    return ("pause", seconds, chunk)


class FakeStream:
    def __init__(self, chunks, delay=0.0):
        # a plain string is a chunk carrying those words
        self.chunks = [words(c) if isinstance(c, str) else c for c in chunks]
        self.delay = delay
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.chunks:
            raise StopAsyncIteration
        chunk = self.chunks.pop(0)
        if isinstance(chunk, tuple) and chunk[0] == "pause":
            await asyncio.sleep(chunk[1])
            chunk = chunk[2]
            chunk = words(chunk) if isinstance(chunk, str) else chunk
        return chunk

    async def close(self):
        self.closed = True


class FakeClient:
    """`client.chat.completions.create(**params)` → the next prepared answer."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **params):
        self.calls.append(params)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _service(primary: FakeClient, fallback: FakeClient, after=0.2, stall=None) -> OpenAILLMService:
    with patch.object(OpenAILLMService, "create_client", return_value=primary):
        service = OpenAILLMService(
            api_key="k",
            settings=OpenAILLMService.Settings(
                model="inference-deepseek-v4-flash",
                extra={"extra_body": {"chat_template_kwargs": {"thinking": False}}},
            ),
            svp_fallback={
                "base_url": "https://fallback.example/v1",
                "api_key": "k2",
                "model": "moonshotai/Kimi-K2.6",
                "extra_body": {"chat_template_kwargs": {"thinking": False}, "x": 1},
                "after_secs": after,
                "stall_secs": stall,
            },
        )
    service._svp_fallback_client = fallback
    return service


PARAMS = {"model": "inference-deepseek-v4-flash", "messages": [], "stream": True,
          "extra_body": {"chat_template_kwargs": {"thinking": False}}}  # fmt: skip


def _said(chunk):
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        return "<tool call>"
    return delta.content


async def _drain(stream) -> list:
    """What the stream carried: the words of each chunk, « <tool call> » for a tool call."""
    return [_said(c) async for c in stream]


@pytest.mark.asyncio
async def test_p46_a_primary_that_answers_in_time_is_kept():
    primary = FakeClient(FakeStream(["a", "b"]))
    fallback = FakeClient()
    service = _service(primary, fallback)
    assert await _drain(await service._svp_completions_with_fallback(dict(PARAMS))) == ["a", "b"]
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_p46_a_slow_first_word_is_asked_again_of_the_fallback_in_the_same_turn():
    slow = FakeStream(["late"], delay=1.0)
    primary = FakeClient(slow)
    fallback = FakeClient(FakeStream(["vite"]))
    service = _service(primary, fallback, after=0.1)
    assert await _drain(await service._svp_completions_with_fallback(dict(PARAMS))) == ["vite"]
    assert slow.closed
    # the fallback's own model and parameters, never the primary's
    sent = fallback.calls[0]
    assert sent["model"] == "moonshotai/Kimi-K2.6"
    assert sent["extra_body"] == {"chat_template_kwargs": {"thinking": False}, "x": 1}
    assert sent["messages"] == PARAMS["messages"]


@pytest.mark.asyncio
async def test_p46_a_refused_request_goes_to_the_fallback():
    refused = APIStatusError(
        "too long", response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
        body=None,
    )  # fmt: skip
    primary = FakeClient(refused)
    fallback = FakeClient(FakeStream(["ok"]))
    service = _service(primary, fallback)
    assert await _drain(await service._svp_completions_with_fallback(dict(PARAMS))) == ["ok"]


@pytest.mark.asyncio
async def test_p46_after_two_failures_the_rest_of_the_call_stays_on_the_fallback():
    primary = FakeClient(FakeStream([], delay=1.0), FakeStream([], delay=1.0), FakeStream(["x"]))
    fallback = FakeClient(FakeStream(["1"]), FakeStream(["2"]), FakeStream(["3"]))
    service = _service(primary, fallback, after=0.05)
    for expected in ("1", "2", "3"):
        assert await _drain(await service._svp_completions_with_fallback(dict(PARAMS))) == [
            expected
        ]
    assert len(primary.calls) == 2  # the third turn never asked the primary


@pytest.mark.asyncio
async def test_p46_one_failure_then_a_good_turn_resets_the_count():
    primary = FakeClient(FakeStream([], delay=1.0), FakeStream(["ok"]), FakeStream([], delay=1.0),
                         FakeStream(["ok2"]))  # fmt: skip
    fallback = FakeClient(FakeStream(["f1"]), FakeStream(["f2"]))
    service = _service(primary, fallback, after=0.05)
    got = [await _drain(await service._svp_completions_with_fallback(dict(PARAMS)))
           for _ in range(4)]  # fmt: skip
    assert got == [["f1"], ["ok"], ["f2"], ["ok2"]]
    assert not service._svp_on_fallback


def test_p46_without_a_fallback_the_service_is_stock():
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(api_key="k")
    assert service._svp_fallback is None


# --- F11 (ADR-078, 2026-11-06): CF-238 and CF-137 - words count, and a stall is bounded -----


@pytest.mark.asyncio
async def test_cf238_a_first_chunk_without_words_is_no_answer_and_the_fallback_takes_over():
    """Run 2195 (2026-10-31): Phoeniqs answered 200 in 0.8 s, then no words for 60 s - the
    first chunk had come, so P-46 kept the primary. An empty chunk no longer counts."""
    silent = FakeStream([empty(), pause(1.0, "trop tard")])
    primary = FakeClient(silent)
    fallback = FakeClient(FakeStream(["Bonjour"]))
    service = _service(primary, fallback, after=0.2)
    assert await _drain(await service._svp_completions_with_fallback(dict(PARAMS))) == ["Bonjour"]
    assert silent.closed and len(fallback.calls) == 1


@pytest.mark.asyncio
async def test_cf238_an_empty_first_chunk_then_words_in_time_keeps_the_primary():
    primary = FakeClient(FakeStream([empty(), "Bonjour", " Madame"]))
    fallback = FakeClient()
    service = _service(primary, fallback, after=0.5)
    assert await _drain(await service._svp_completions_with_fallback(dict(PARAMS))) == [
        "", "Bonjour", " Madame",
    ]  # fmt: skip
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_cf238_a_tool_call_first_is_an_answer():
    primary = FakeClient(FakeStream([empty(), tool_call()]))
    fallback = FakeClient()
    service = _service(primary, fallback, after=0.5)
    assert await _drain(await service._svp_completions_with_fallback(dict(PARAMS))) == [
        "", "<tool call>",
    ]  # fmt: skip
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_cf137_a_stall_after_the_first_words_ends_the_turn_with_what_was_said():
    """Words were pushed: never a retry after them; the turn ends at the stall's limit and
    the stream is closed - the caller is not left in a minute of silence."""
    stalled = FakeStream(["Je regarde", pause(5.0, " jamais")])
    primary = FakeClient(stalled)
    fallback = FakeClient()
    service = _service(primary, fallback, after=0.5, stall=0.2)
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await _drain(await service._svp_completions_with_fallback(dict(PARAMS))) == ["Je regarde"]
    assert loop.time() - started < 1.0
    assert stalled.closed and fallback.calls == []


@pytest.mark.asyncio
async def test_cf137_the_fallbacks_own_stream_is_watched_too():
    primary = FakeClient(FakeStream([], delay=1.0))
    stalled = FakeStream(["Un instant", pause(5.0, " jamais")])
    fallback = FakeClient(stalled)
    service = _service(primary, fallback, after=0.05, stall=0.2)
    assert await _drain(await service._svp_completions_with_fallback(dict(PARAMS))) == ["Un instant"]
    assert stalled.closed


@pytest.mark.asyncio
async def test_cf137_without_a_stall_limit_the_stream_is_read_to_its_end():
    """`stall_secs` None (SVP_LLM_STREAM_STALL_MS unset): no watch between chunks - stock."""
    primary = FakeClient(FakeStream(["a", pause(0.3, "b")]))
    service = _service(primary, FakeClient(), after=0.5, stall=None)
    assert await _drain(await service._svp_completions_with_fallback(dict(PARAMS))) == ["a", "b"]
