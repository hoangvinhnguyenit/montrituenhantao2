import sys
import types
import unittest
from unittest.mock import patch

import chatbot


class ChatbotProviderTests(unittest.TestCase):
    def test_gemini_key_takes_precedence_over_anthropic(self):
        with patch.object(
            chatbot,
            "_secret",
            side_effect=lambda name: {
                "GEMINI_API_KEY": "google-test-key",
                "ANTHROPIC_API_KEY": "anthropic-test-key",
            }.get(name),
        ):
            self.assertEqual(chatbot._configured_provider(), "gemini")

    def test_gemini_stream_uses_configured_key_model_and_history(self):
        captured = {}

        class FakeModels:
            def generate_content_stream(self, **kwargs):
                captured.update(kwargs)
                return [types.SimpleNamespace(text="Xin "), types.SimpleNamespace(text="chào")]

        fake_genai = types.ModuleType("google.genai")
        fake_genai.Client = lambda api_key: (
            captured.update(api_key=api_key) or types.SimpleNamespace(models=FakeModels())
        )
        fake_genai.types = types.SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: kwargs
        )
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai
        secrets = {
            "GEMINI_API_KEY": "google-test-key",
            "GEMINI_MODEL": "gemini-test-model",
        }

        with (
            patch.dict(
                sys.modules,
                {"google": fake_google, "google.genai": fake_genai},
            ),
            patch.object(chatbot, "_secret", side_effect=secrets.get),
        ):
            answer = list(
                chatbot._stream_ai(
                    "Câu hỏi mới",
                    "số liệu báo cáo",
                    [{"role": "assistant", "content": "Câu trả lời cũ"}],
                )
            )

        self.assertEqual(answer, ["Xin ", "chào"])
        self.assertEqual(captured["api_key"], "google-test-key")
        self.assertEqual(captured["model"], "gemini-test-model")
        self.assertEqual(
            captured["contents"],
            [
                {"role": "model", "parts": [{"text": "Câu trả lời cũ"}]},
                {"role": "user", "parts": [{"text": "Câu hỏi mới"}]},
            ],
        )
        self.assertIn("số liệu báo cáo", captured["config"]["system_instruction"])

    def test_gemini_503_falls_back_to_stable_model_before_output(self):
        requested_models = []

        class FakeModels:
            def generate_content_stream(self, **kwargs):
                requested_models.append(kwargs["model"])
                if len(requested_models) == 1:
                    error = RuntimeError("temporarily unavailable")
                    error.code = 503
                    raise error
                return [types.SimpleNamespace(text="Đã trả lời")]

        fake_genai = types.ModuleType("google.genai")
        fake_genai.Client = lambda api_key: types.SimpleNamespace(models=FakeModels())
        fake_genai.types = types.SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: kwargs
        )
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai

        with (
            patch.dict(
                sys.modules,
                {"google": fake_google, "google.genai": fake_genai},
            ),
            patch.object(
                chatbot,
                "_secret",
                side_effect=lambda name: {"GEMINI_API_KEY": "google-test-key"}.get(name),
            ),
        ):
            answer = list(chatbot._stream_ai("hỏi", "context", []))

        self.assertEqual(answer, ["Đã trả lời"])
        self.assertEqual(
            requested_models,
            [chatbot.DEFAULT_GEMINI_MODEL, chatbot.FALLBACK_GEMINI_MODEL],
        )

    def test_gemini_server_error_suggests_retry_without_showing_key(self):
        class ServerError(Exception):
            code = 503

        with patch.object(
            chatbot,
            "_configured_provider",
            return_value="gemini",
        ):
            message = "".join(chatbot._stream_error(ServerError("internal details")))

        self.assertIn("HTTP 503", message)
        self.assertIn("Thử gửi lại", message)
        self.assertNotIn("internal details", message)

    def test_gemini_quota_error_points_to_quota_settings(self):
        class ClientError(Exception):
            code = 429

        with patch.object(
            chatbot,
            "_configured_provider",
            return_value="gemini",
        ):
            message = "".join(chatbot._stream_error(ClientError()))

        self.assertIn("HTTP 429", message)
        self.assertIn("quota/billing", message)


if __name__ == "__main__":
    unittest.main()
