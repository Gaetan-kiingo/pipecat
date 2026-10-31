#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for node-transition function-call deferral in OpenAI LLM services."""

from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.llm import OpenAILLMService


def _make_service() -> OpenAILLMService:
    with patch.object(OpenAILLMService, "create_client"):
        return OpenAILLMService(api_key="test-key")


def _make_function_call(name: str, tool_call_id: str) -> FunctionCallFromLLM:
    return FunctionCallFromLLM(
        context=LLMContext(),
        tool_call_id=tool_call_id,
        function_name=name,
        arguments={},
    )


@pytest.mark.asyncio
async def test_non_transition_call_is_not_deferred_after_generated_text():
    service = _make_service()
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call("look_up_account", "call-ordinary")

    await service._run_or_defer_function_calls(
        [function_call],
        text_generated=True,
    )

    service.run_function_calls.assert_awaited_once_with([function_call])
    assert service._pending_node_transition_function_calls == []


@pytest.mark.asyncio
async def test_node_transition_call_is_deferred_after_generated_text():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call(
        "transition_to_next_node",
        "call-transition",
    )

    await service._run_or_defer_function_calls(
        [function_call],
        text_generated=True,
    )

    service.run_function_calls.assert_not_awaited()
    assert service._pending_node_transition_function_calls == [function_call]


@pytest.mark.asyncio
async def test_node_transition_call_runs_immediately_without_generated_text():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call(
        "transition_to_next_node",
        "call-transition",
    )

    await service._run_or_defer_function_calls(
        [function_call],
        text_generated=False,
    )

    service.run_function_calls.assert_awaited_once_with([function_call])
    assert service._pending_node_transition_function_calls == []


@pytest.mark.asyncio
async def test_mixed_batch_with_node_transition_is_deferred_together():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    function_calls = [
        _make_function_call("look_up_account", "call-ordinary"),
        _make_function_call("transition_to_next_node", "call-transition"),
    ]

    await service._run_or_defer_function_calls(
        function_calls,
        text_generated=True,
    )

    service.run_function_calls.assert_not_awaited()
    assert service._pending_node_transition_function_calls == function_calls


@pytest.mark.asyncio
async def test_pending_node_transition_batch_runs_after_tts():
    service = _make_service()
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call(
        "transition_to_next_node",
        "call-transition",
    )
    service._pending_node_transition_function_calls = [function_call]

    await service._run_pending_node_transition_function_calls()

    service.run_function_calls.assert_awaited_once_with([function_call])
    assert service._pending_node_transition_function_calls == []


# --- P-45 (Swiss Voice Platform): blank text before a tool call is not speech ---------------


def _chunk(content=None, tool_name=None):
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import (
        Choice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    tool_calls = None
    if tool_name:
        tool_calls = [
            ChoiceDeltaToolCall(
                index=0,
                id="call-1",
                type="function",
                function=ChoiceDeltaToolCallFunction(name=tool_name, arguments="{}"),
            )
        ]
    return ChatCompletionChunk(
        id="c",
        object="chat.completion.chunk",
        created=0,
        model="inference-deepseek-v32",
        choices=[Choice(index=0, delta=ChoiceDelta(content=content, tool_calls=tool_calls))],
    )


async def _stream_signal(monkeypatch, switch: str | None, chunks) -> bool:
    """What `_process_context` tells the deferral about this reply's text."""
    if switch is None:
        monkeypatch.delenv("SVP_SPOKEN_TEXT_ONLY", raising=False)
    else:
        monkeypatch.setenv("SVP_SPOKEN_TEXT_ONLY", switch)
    service = _make_service()

    async def stream():
        for c in chunks:
            yield c

    service.get_chat_completions = AsyncMock(return_value=stream())
    service._push_llm_text = AsyncMock()
    service.push_frame = AsyncMock()
    seen = {}

    async def capture(function_calls, *, text_generated):
        seen["text_generated"] = text_generated

    service._run_or_defer_function_calls = capture
    await service._process_context(LLMContext())
    return seen["text_generated"]


@pytest.mark.asyncio
async def test_p45_a_blank_line_before_a_step_change_is_not_speech(monkeypatch):
    # DeepSeek V3.2 / V4 Flash at Phoeniqs, 2026-10-30: « \n\n » then the tool call
    chunks = [_chunk(content="\n\n"), _chunk(tool_name="vers_rendez_vous")]
    assert await _stream_signal(monkeypatch, "on", chunks) is False
    # stock: the blank line counted, the transition waited for a voice that never spoke
    assert await _stream_signal(monkeypatch, None, chunks) is True


@pytest.mark.asyncio
async def test_p45_words_before_a_step_change_still_wait_for_the_voice(monkeypatch):
    chunks = [_chunk(content="\n"), _chunk(content="Un instant."), _chunk(tool_name="vers_fin")]
    assert await _stream_signal(monkeypatch, "on", chunks) is True
