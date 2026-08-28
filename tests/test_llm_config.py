import logging

from agent_framework.llm.config import LLMConfig


def test_config_never_logs_api_keys(monkeypatch, caplog):
    secret = "sk-test-secret-that-must-not-appear"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)

    with caplog.at_level(logging.INFO):
        LLMConfig(model_name="openrouter/z-ai/glm-5.3-flash")

    assert secret not in caplog.text
    assert "DEEPSEEK_API_KEY" not in caplog.text
