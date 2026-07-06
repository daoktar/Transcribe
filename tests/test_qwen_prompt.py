"""Tests for the file-backed Qwen context prompt (transcribe.qwen_prompt)."""
from transcribe import qwen_prompt


def test_bundled_prompt_loads_and_has_domain_content():
    text = qwen_prompt.load_context_prompt()
    assert isinstance(text, str)
    assert len(text) > 100
    # The code-switch instruction is the whole point of the prompt.
    assert "латиниц" in text
    # kontur.talk correction is reflected (and the толк→Mattermost mis-mapping is gone).
    assert "kontur.talk" in text


def test_env_override_wins(tmp_path, monkeypatch):
    custom = tmp_path / "custom_prompt.txt"
    custom.write_text("МОЙ СОБСТВЕННЫЙ КОНТЕКСТ", encoding="utf-8")
    monkeypatch.setenv("TRANSCRIBE_QWEN_PROMPT_FILE", str(custom))
    assert qwen_prompt.context_prompt_path() == custom
    assert qwen_prompt.load_context_prompt() == "МОЙ СОБСТВЕННЫЙ КОНТЕКСТ"


def test_fallback_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_QWEN_PROMPT_FILE", str(tmp_path / "does_not_exist.txt"))
    out = qwen_prompt.load_context_prompt()
    assert out == qwen_prompt._FALLBACK_PROMPT
    assert "латиниц" in out


def test_default_attribute_matches_loader():
    # Backwards-compatible module attribute mirrors the file content.
    assert qwen_prompt.DEFAULT_CONTEXT_PROMPT == qwen_prompt.load_context_prompt()
