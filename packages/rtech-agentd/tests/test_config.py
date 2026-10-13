"""Agent config: shape parsing, ref indirection, literal-secret rejection."""

from __future__ import annotations

from pathlib import Path

import pytest
from rtech_agentd.config import ConfigError, load_agent_config

_GOOD = """
agent: nova
home: /home/nova/.rtech-agents
engine:
  kind: claude
  model: "claude-sonnet-5"
  litellm_base: "http://litellm.example.internal:4000/v1"
  backup_model: "claude-haiku-4-5"
nats:
  url: "nats://127.0.0.1:4222"
  env: dev
openbrain:
  base_url: "http://openbrain.example.internal:3100"
  token_ref: "vaultwarden:ob-nova-token"
discord:
  bot_token_ref: "vaultwarden:discord-nova"
  home_channel: "123"
  free_response_channels: ["456", "789"]
  allow_bots: mentions
persona:
  soul: SOUL.md
  user: USER.md
  kernel: KERNEL.md
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "agent.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_good_config_parses(tmp_path: Path) -> None:
    cfg = load_agent_config(_write(tmp_path, _GOOD))
    assert cfg.agent == "nova"
    assert cfg.engine.kind == "claude"
    assert cfg.engine.litellm_base == "http://litellm.example.internal:4000/v1"
    assert cfg.nats.env == "dev"
    assert cfg.openbrain.token_ref == "vaultwarden:ob-nova-token"
    assert cfg.discord.free_response_channels == ("456", "789")
    assert cfg.persona.soul == "SOUL.md"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_agent_config(tmp_path / "nope.yaml")


def test_missing_required_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_agent_config(_write(tmp_path, "agent: nova\nhome: /h\n"))  # no engine/nats


def test_literal_secret_in_ref_is_rejected(tmp_path: Path) -> None:
    body = _GOOD.replace(
        'token_ref: "vaultwarden:ob-nova-token"',
        'token_ref: "sk-liveSECRETtokenValue0123456789abcdef"',
    )
    with pytest.raises(ConfigError) as exc:
        load_agent_config(_write(tmp_path, body))
    # The error must NOT echo the secret value.
    assert "sk-liveSECRET" not in str(exc.value)


def test_env_ref_provider_allowed(tmp_path: Path) -> None:
    body = _GOOD.replace('token_ref: "vaultwarden:ob-nova-token"', 'token_ref: "env:OB_TOKEN_NOVA"')
    cfg = load_agent_config(_write(tmp_path, body))
    assert cfg.openbrain.token_ref == "env:OB_TOKEN_NOVA"
