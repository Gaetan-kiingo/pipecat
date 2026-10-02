"""P-28 (Swiss Voice Platform): what Azure says each sentence cost."""

import unittest
from unittest.mock import patch

from pipecat.services.azure import svp_voice_timing


class _Properties:
    def __init__(self, values):
        self._values = values

    def get_property(self, property_id):
        return self._values.get(property_id.name, "")


class _Result:
    def __init__(self, values):
        self.properties = _Properties(values)


class TestAzureLatencies(unittest.TestCase):
    def test_the_switch_is_the_timing_marks_switch(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(svp_voice_timing.enabled())
        with patch.dict("os.environ", {"SVP_TURN_TIMING": "on"}):
            self.assertTrue(svp_voice_timing.enabled())

    def test_azures_figures_are_read_as_milliseconds(self):
        result = _Result(
            {
                "SpeechServiceResponse_SynthesisConnectionLatencyMs": "312",
                "SpeechServiceResponse_SynthesisNetworkLatencyMs": "21",
                "SpeechServiceResponse_SynthesisServiceLatencyMs": "35.0",
                "SpeechServiceResponse_SynthesisFirstByteLatencyMs": "368",
                "SpeechServiceResponse_SynthesisFinishLatencyMs": "702",
            }
        )
        self.assertEqual(
            svp_voice_timing.azure_latencies(result),
            {
                "connection_ms": 312,
                "network_ms": 21,
                "service_ms": 35,
                "first_byte_ms": 368,
                "finish_ms": 702,
            },
        )

    def test_a_figure_azure_did_not_give_is_left_out(self):
        result = _Result({"SpeechServiceResponse_SynthesisFirstByteLatencyMs": "48"})
        self.assertEqual(svp_voice_timing.azure_latencies(result), {"first_byte_ms": 48})
        self.assertEqual(svp_voice_timing.azure_latencies(object()), {})
        garbled = _Result({"SpeechServiceResponse_SynthesisFirstByteLatencyMs": "soon"})
        self.assertEqual(svp_voice_timing.azure_latencies(garbled), {})
