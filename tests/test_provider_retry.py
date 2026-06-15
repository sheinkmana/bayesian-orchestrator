from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bayesian_orchestrator.workflows.mmlu_bayesian_orchestrator import (
    MMLUQuestion,
    ModelSpec,
    OpenAICompatibleProvider,
)


class _StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _response():
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"A","confidence":0.9}'))],
    )


class ProviderRetryTests(unittest.TestCase):
    @patch("openai.OpenAI")
    @patch("time.sleep")
    def test_transient_502_is_retried(self, sleep: Mock, openai: Mock) -> None:
        client = openai.return_value
        client.chat.completions.create.side_effect = [_StatusError(502), _response()]
        with patch.dict("os.environ", {"TEST_API_KEY": "secret"}):
            provider = OpenAICompatibleProvider(
                "https://example.test/v1/",
                "TEST_API_KEY",
                0.0,
                32,
                True,
                request_max_attempts=3,
                retry_initial_delay_seconds=0.25,
                retry_max_delay_seconds=1.0,
            )

        result = provider.call(
            "Question",
            ModelSpec("small", "provider/model", 0.01),
            MMLUQuestion("q1", "subject", "Question", ("One", "Two"), "A"),
        )

        self.assertEqual(result["completion_tokens"], 4)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        sleep.assert_called_once_with(0.25)

    @patch("openai.OpenAI")
    @patch("time.sleep")
    def test_non_retryable_error_is_raised_immediately(self, sleep: Mock, openai: Mock) -> None:
        client = openai.return_value
        client.chat.completions.create.side_effect = _StatusError(401)
        with patch.dict("os.environ", {"TEST_API_KEY": "secret"}):
            provider = OpenAICompatibleProvider(
                "https://example.test/v1/", "TEST_API_KEY", 0.0, 32, False
            )

        with self.assertRaises(_StatusError):
            provider.call(
                "Question",
                ModelSpec("small", "provider/model", 0.01),
                MMLUQuestion("q1", "subject", "Question", ("One", "Two"), "A"),
            )

        self.assertEqual(client.chat.completions.create.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
