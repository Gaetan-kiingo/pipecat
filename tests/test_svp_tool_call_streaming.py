"""P-53 (Swiss Voice Platform, ADR-002): the pieces of a tool call are announced as they stream,
so a listener (the runtime's booking sentence, P-52) can act before the call is complete."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService


def piece(name=None, arguments=None, call_id=None):
    call = SimpleNamespace(
        index=0, id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[call], audio=None))
        ],
    )


class Stream:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

    async def close(self):
        pass


BOOKING = [
    piece(name="book_appointment", arguments="", call_id="call-7"),
    piece(arguments='{"confirm": "p1'),
    piece(arguments='.k1.176254.abc.def", "date": "2026-11-11"'),
    piece(arguments=', "time": "16:00"}'),
]


def _service():
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(api_key="test-key")
    service.get_chat_completions = AsyncMock(return_value=Stream(BOOKING))
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.push_frame = AsyncMock()
    service._run_or_defer_function_calls = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_p53_the_pieces_of_a_tool_call_are_announced_as_they_stream(monkeypatch):
    monkeypatch.setenv("SVP_WRITE_PROGRESS_EARLY", "on")
    service = _service()
    seen = []

    async def listener(_service, name, arguments, call_id):
        seen.append((name, arguments, call_id))

    service.add_event_handler("on_svp_tool_call_streaming", listener)
    await service._process_context(MagicMock(spec=LLMContext))
    assert [s[0] for s in seen] == ["book_appointment"] * 4
    assert seen[1] == ("book_appointment", '{"confirm": "p1', "call-7")
    # the listener heard the confirmation begin before the call was complete
    assert seen[-1][1].endswith('"16:00"}')
    service._run_or_defer_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_p53_unset_is_stock_no_event(monkeypatch):
    monkeypatch.delenv("SVP_WRITE_PROGRESS_EARLY", raising=False)
    service = _service()
    seen = []

    async def listener(*args):
        seen.append(args)

    service.add_event_handler("on_svp_tool_call_streaming", listener)
    await service._process_context(MagicMock(spec=LLMContext))
    assert seen == []
    service._run_or_defer_function_calls.assert_awaited_once()
