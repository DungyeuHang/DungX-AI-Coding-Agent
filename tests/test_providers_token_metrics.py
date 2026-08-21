from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from local_agent.config import AgentConfig
from local_agent.models import ProjectContext, ProviderMetric
from local_agent.providers import (
    AnthropicProvider,
    BaseHTTPProvider,
    MockProvider,
    build_provider,
)


class ProviderTokenMetricsTests(unittest.TestCase):
    def test_extract_token_counts_openai_shape(self):
        payload = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        inp, out = BaseHTTPProvider._extract_token_counts(payload)
        self.assertEqual(inp, 10)
        self.assertEqual(out, 5)

    def test_extract_token_counts_gemini_shape(self):
        payload = {"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 8}}
        inp, out = BaseHTTPProvider._extract_token_counts(payload)
        self.assertEqual(inp, 20)
        self.assertEqual(out, 8)

    def test_extract_token_counts_missing_returns_none(self):
        payload = {}
        inp, out = BaseHTTPProvider._extract_token_counts(payload)
        self.assertIsNone(inp)
        self.assertIsNone(out)

    def test_extract_token_counts_non_dict_payload(self):
        payload = [1, 2]
        inp, out = BaseHTTPProvider._extract_token_counts(payload)
        self.assertIsNone(inp)
        self.assertIsNone(out)

    def test_extract_token_counts_zero_values(self):
        payload = {"usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        inp, out = BaseHTTPProvider._extract_token_counts(payload)
        self.assertEqual(inp, 0)
        self.assertEqual(out, 0)

    def test_record_metric_stores_actual_tokens(self):
        provider = MockProvider()
        provider.metrics_enabled = True
        provider._record_metric(
            request_type="test_req",
            input_size=40,
            output_size=20,
            duration_seconds=0.1,
            model="mock",
            succeeded=True,
            actual_input_tokens=10,
            actual_output_tokens=5,
        )
        self.assertEqual(len(provider.provider_metrics), 1)
        metric = provider.provider_metrics[0]
        self.assertEqual(metric.actual_input_tokens, 10)
        self.assertEqual(metric.actual_output_tokens, 5)

    def test_record_metric_char4_fallback_when_none(self):
        provider = MockProvider()
        provider.metrics_enabled = True
        provider._record_metric(
            request_type="test_req",
            input_size=40,
            output_size=20,
            duration_seconds=0.1,
            model="mock",
            succeeded=True,
            actual_input_tokens=None,
            actual_output_tokens=None,
        )
        self.assertEqual(len(provider.provider_metrics), 1)
        metric = provider.provider_metrics[0]
        self.assertIsNone(metric.actual_input_tokens)
        self.assertIsNone(metric.actual_output_tokens)
        self.assertGreater(metric.approximate_input_tokens, 0)
        self.assertGreater(metric.approximate_output_tokens, 0)

    def test_mock_provider_emits_token_metrics(self):
        provider = MockProvider()
        provider.metrics_enabled = True
        context = ProjectContext(root=Path("."))
        provider.generate_plan("Test task", context)
        self.assertEqual(len(provider.provider_metrics), 1)
        metric = provider.provider_metrics[0]
        self.assertEqual(metric.actual_input_tokens, 10)
        self.assertEqual(metric.actual_output_tokens, 5)

    def test_build_provider_anthropic_stub_returns_instance(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"}, clear=False):
            config = AgentConfig(project=Path("."), provider="anthropic")
            provider = build_provider(config)
            self.assertIsInstance(provider, AnthropicProvider)

    def test_provider_metric_round_trips_with_new_fields(self):
        metric = ProviderMetric(
            request_type="generate_plan",
            input_size=40,
            output_size=20,
            model="test-model",
            duration_seconds=0.123,
            succeeded=True,
            actual_input_tokens=10,
            actual_output_tokens=5,
        )
        d = metric.to_dict()
        restored = ProviderMetric.from_dict(d)
        self.assertEqual(restored.actual_input_tokens, 10)
        self.assertEqual(restored.actual_output_tokens, 5)

    def test_provider_metric_from_dict_missing_fields_defaults_to_none(self):
        old_dict = {
            "request_type": "generate_plan",
            "input_size": 40,
            "output_size": 20,
            "model": "test-model",
            "duration_seconds": 0.123,
            "succeeded": True,
        }
        restored = ProviderMetric.from_dict(old_dict)
        self.assertIsNone(restored.actual_input_tokens)
        self.assertIsNone(restored.actual_output_tokens)


if __name__ == "__main__":
    unittest.main()

