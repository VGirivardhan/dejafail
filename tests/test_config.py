import pytest

from dejafail.config import ConfigError, load_config


def test_missing_key_is_named_but_value_never_shown():
    with pytest.raises(ConfigError) as exc:
        load_config({"HINDSIGHT_API_KEY": "hsk_secret_value"})
    assert "GROQ_API_KEY" in str(exc.value)
    assert "hsk_secret_value" not in str(exc.value)


def test_whitespace_only_key_counts_as_missing():
    with pytest.raises(ConfigError):
        load_config({"HINDSIGHT_API_KEY": "   ", "GROQ_API_KEY": "gsk_y"})


def test_defaults_and_bank_id():
    cfg = load_config({"HINDSIGHT_API_KEY": "hsk_x", "GROQ_API_KEY": "gsk_y"})
    assert cfg.hindsight_url == "https://api.hindsight.vectorize.io"
    assert cfg.groq_model == "openai/gpt-oss-120b"
    assert cfg.repo == "shopfront"
    assert cfg.bank_id == "dejafail-shopfront"


def test_repr_hides_keys():
    cfg = load_config({"HINDSIGHT_API_KEY": "hsk_x", "GROQ_API_KEY": "gsk_y"})
    assert "hsk_x" not in repr(cfg)
    assert "gsk_y" not in repr(cfg)
