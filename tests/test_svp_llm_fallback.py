"""P-46 (Swiss Voice Platform, ADR-065): the in-call fallback of an OpenAI-compatible engine.

The primary engine has ``after_secs`` to give its first chunk; a failure or a silence until
then sends the same request to the fallback in the same turn, with the fallback's own model
and extra parameters; after two such turns the rest of the call stays on the fallback."""

import asyncio
from unittest.mock import patch

import httpx
import pytest
from openai import APIStatusError

from pipecat.services.openai.llm import OpenAILLMService


class FakeStream:
    def __init__(self, chunks, delay=0.0):
        self.chunks = list(chunks)
        self.delay = delay
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

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


def _service(primary: FakeClient, fallback: FakeClient, after=0.2) -> OpenAILLMService:
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
            },
        )
    service._svp_fallback_client = fallback
    return service


PARAMS = {"model": "inference-deepseek-v4-flash", "messages": [], "stream": True,
          "extra_body": {"chat_template_kwargs": {"thinking": False}}}  # fmt: skip


async def _drain(stream) -> list:
    return [c async for c in stream]


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
