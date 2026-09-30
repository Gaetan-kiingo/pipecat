"""P-26 (Swiss Voice Platform, ADR-002): the per-turn part of the system instruction is
sent after the history when SVP_TRAILING_CONTEXT names the section header; stock
behaviour otherwise."""

import pytest

from pipecat.adapters.services.open_ai_adapter import _svp_place_system_instruction

INSTRUCTION = "[ROLE]\nYou are the receptionist.\n\n[NOW]\nIt is {{now}}.\n\n[LANGUAGE]\nFrench."
HISTORY = [
    {"role": "user", "content": "Bonjour"},
    {"role": "assistant", "content": "Bonjour, que puis-je faire ?"},
    {"role": "user", "content": "Vos horaires ?"},
]


def test_stock_when_unset(monkeypatch):
    monkeypatch.delenv("SVP_TRAILING_CONTEXT", raising=False)
    out = _svp_place_system_instruction(INSTRUCTION, HISTORY)
    assert out[0] == {"role": "system", "content": INSTRUCTION}
    assert out[1:] == HISTORY


def test_per_turn_section_goes_before_the_last_user_message(monkeypatch):
    monkeypatch.setenv("SVP_TRAILING_CONTEXT", "[NOW]")
    out = _svp_place_system_instruction(INSTRUCTION, HISTORY)
    assert out[0] == {"role": "system", "content": "[ROLE]\nYou are the receptionist."}
    assert out[1:3] == HISTORY[:2]
    assert out[3] == {"role": "system", "content": "[NOW]\nIt is {{now}}.\n\n[LANGUAGE]\nFrench."}
    assert out[4] == HISTORY[2]  # the request still ends with the caller's turn


def test_tool_result_at_the_end_is_not_separated_from_its_call(monkeypatch):
    monkeypatch.setenv("SVP_TRAILING_CONTEXT", "[NOW]")
    history = HISTORY + [
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]
    out = _svp_place_system_instruction(INSTRUCTION, history)
    roles = [m["role"] for m in out]
    assert roles == ["system", "user", "assistant", "system", "user", "assistant", "tool"]


@pytest.mark.parametrize("history", [[], [{"role": "assistant", "content": "Bonjour"}]])
def test_whole_instruction_until_the_caller_has_spoken(monkeypatch, history):
    monkeypatch.setenv("SVP_TRAILING_CONTEXT", "[NOW]")
    out = _svp_place_system_instruction(INSTRUCTION, history)
    assert out[0] == {"role": "system", "content": INSTRUCTION}


def test_header_absent_means_stock(monkeypatch):
    monkeypatch.setenv("SVP_TRAILING_CONTEXT", "[PER TURN]")
    out = _svp_place_system_instruction(INSTRUCTION, HISTORY)
    assert out[0]["content"] == INSTRUCTION and len(out) == 4


def test_header_inside_a_line_is_not_a_split_point(monkeypatch):
    monkeypatch.setenv("SVP_TRAILING_CONTEXT", "[NOW]")
    instruction = "[FACTS]\nUse [NOW] to say whether the business is open.\n\n[LANGUAGE]\nFrench."
    out = _svp_place_system_instruction(instruction, HISTORY)
    assert out[0]["content"] == instruction and len(out) == 4
