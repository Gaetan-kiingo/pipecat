#
# Swiss Voice Platform (ADR-002, P-28 - the timing marks; on the experiment branch).
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""What Azure says each sentence cost, for the run's timeline.

The voice's first audio comes in about 0.05 s on some turns and in 0.4 to 0.7 s on
others. Azure's library reports, for every sentence, how long the connection, the network
and the service itself took; with ``SVP_TURN_TIMING=on`` the voice service puts those
figures and the state of its connection on the timeline - numbers and booleans, never
the sentence.
"""

from __future__ import annotations

import os
from typing import Any

MARK = "voice"
CONNECTION_MARK = "voice_connection"

# the mark's field -> the name of Azure's property
LATENCIES = {
    "connection_ms": "SpeechServiceResponse_SynthesisConnectionLatencyMs",
    "network_ms": "SpeechServiceResponse_SynthesisNetworkLatencyMs",
    "service_ms": "SpeechServiceResponse_SynthesisServiceLatencyMs",
    "first_byte_ms": "SpeechServiceResponse_SynthesisFirstByteLatencyMs",
    "finish_ms": "SpeechServiceResponse_SynthesisFinishLatencyMs",
}


def enabled() -> bool:
    """Whether the timing marks are kept (the switch of P-28)."""
    return os.getenv("SVP_TURN_TIMING", "").strip().lower() == "on"


def azure_latencies(result: Any) -> dict[str, int]:
    """Azure's own figures for one synthesis result, in milliseconds.

    A figure Azure did not give is left out rather than guessed.
    """
    from azure.cognitiveservices.speech import PropertyId

    out: dict[str, int] = {}
    properties = getattr(result, "properties", None)
    if properties is None:
        return out
    for field, name in LATENCIES.items():
        try:
            raw = properties.get_property(getattr(PropertyId, name))
            if raw not in (None, ""):
                out[field] = int(float(raw))
        except Exception:
            continue
    return out
