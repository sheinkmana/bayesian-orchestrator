from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bayesian_orchestrator.workflows.mmlu_bayesian_orchestrator import (
    CallResult,
    ModelSpec,
    _observed_call_cost_usd,
    _operational_metrics,
    _resolve_model_specs,
    _validate_available_models,
)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class PricingTests(unittest.TestCase):
    def test_catalog_rates_are_resolved_from_per_million_to_per_thousand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "pricing.yaml"
            catalog.write_text(
                """
version: test-v1
as_of: 2026-06-14
currency: USD
unit: per_1m_tokens
models:
  provider/model:
    input: 0.10
    output: 0.30
""".strip(),
                encoding="utf-8",
            )
            specs, snapshot = _resolve_model_specs(
                {
                    "pricing": {"catalog": str(catalog), "required": True},
                    "models": [{"label": "small", "model_id": "provider/model", "cost_weight": 0.01}],
                }
            )

        self.assertAlmostEqual(specs[0].input_cost_per_1k or 0.0, 0.00010)
        self.assertAlmostEqual(specs[0].output_cost_per_1k or 0.0, 0.00030)
        self.assertEqual(snapshot.version, "test-v1")
        self.assertEqual(snapshot.prices_per_1m["provider/model"], {"input": 0.10, "output": 0.30})

    def test_catalog_requires_every_selected_model_price(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "pricing.yaml"
            catalog.write_text("unit: per_1m_tokens\nmodels: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "No complete pricing entry"):
                _resolve_model_specs(
                    {
                        "pricing": {"catalog": str(catalog), "required": True},
                        "models": [{"label": "missing", "model_id": "provider/missing"}],
                    }
                )

    def test_provider_model_validation_uses_verbose_models_endpoint(self) -> None:
        payload = json.dumps({"data": [{"id": "provider/model"}]}).encode()
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.headers.get("Authorization")
            captured["timeout"] = timeout
            return _Response(payload)

        with patch.dict(os.environ, {"TEST_API_KEY": "secret"}), patch(
            "urllib.request.urlopen", side_effect=fake_urlopen
        ):
            _validate_available_models(
                {
                    "type": "nebius",
                    "base_url": "https://example.test/v1/",
                    "api_key_env": "TEST_API_KEY",
                },
                [ModelSpec("small", "provider/model", 0.01)],
            )

        self.assertEqual(captured["url"], "https://example.test/v1/models?verbose=true")
        self.assertEqual(captured["authorization"], "Bearer secret")

    def test_operational_metrics_use_observed_usage_and_latency(self) -> None:
        spec = ModelSpec("small", "provider/model", 0.01, input_cost_per_1k=0.0001, output_cost_per_1k=0.0003)
        row = CallResult(
            question_id="q1",
            subject="subject",
            question="Question?",
            choices=("One", "Two"),
            gold_answer="A",
            model_label="small",
            model_id="provider/model",
            cost_weight=0.01,
            question_length_bucket=0,
            answer="A",
            confidence=0.9,
            correct=1,
            latency_ms=2000.0,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            json_mode_requested=True,
            json_mode_used=True,
            raw_response='{"answer":"A","confidence":0.9}',
        )
        metrics = _operational_metrics([row], {"small": spec})
        self.assertAlmostEqual(metrics["small observed total cost"], 0.000016)
        self.assertEqual(metrics["small effective output tokens per second p50"], 10.0)

    def test_observed_usd_cost_never_falls_back_to_cost_weight(self) -> None:
        row = CallResult(
            question_id="q1",
            subject="subject",
            question="Question?",
            choices=("One", "Two"),
            gold_answer="A",
            model_label="small",
            model_id="provider/model",
            cost_weight=99.0,
            question_length_bucket=0,
            answer="A",
            confidence=0.9,
            correct=1,
            latency_ms=100.0,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            json_mode_requested=True,
            json_mode_used=True,
            raw_response='{"answer":"A","confidence":0.9}',
        )
        unpriced = ModelSpec("small", "provider/model", 99.0)
        priced = ModelSpec("small", "provider/model", 99.0, input_cost_per_1k=0.0001, output_cost_per_1k=0.0003)

        self.assertIsNone(_observed_call_cost_usd(row, {"small": unpriced}))
        self.assertAlmostEqual(_observed_call_cost_usd(row, {"small": priced}) or 0.0, 0.000016)


if __name__ == "__main__":
    unittest.main()
