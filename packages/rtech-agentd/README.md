# rtech-agentd

Per-agent supervisor daemon. Owns ONE persistent, inline-steerable engine session
behind an `EngineSession` protocol (Claude Agent SDK v1), consumes typed
`rtech_frames` `TaskFrame`s from NATS, and publishes `TurnFrame`s back. Done is
orchestrator-derived from the frame graph — agentd never emits it.

Implements spec **phases 2-3** (managed engine session, NATS ingress, walking
skeleton).

## Modules

| Module | Role |
|--------|------|
| `engine.py` | `EngineSession` protocol + typed `EngineEvent` union (`text_delta`/`tool_use`/`turn_done`/`error`) + `ToolsPolicy`. SDK-free. |
| `claude_engine.py` | `ClaudeEngine`: `EngineSession` over the Claude Agent SDK `ClaudeSDKClient` in streaming-input mode. Persistent session; mid-turn `send` = inline steering; `interrupt`; session-id capture for `resume`; model/base-url from config (LiteLLM-compatible via `ANTHROPIC_BASE_URL`). |
| `supervisor.py` | Owns the session: resume from an atomic on-disk JSON registry, heartbeat publisher on `agent_heartbeat`, stall watchdog, and the single event pump (fans out to registered sinks). |
| `nats_ingress.py` | Typed-frame edge. Subscribes `direct` + role lanes (bare **and** wildcard dispatch). Inbound envelopes are lifted to a `TaskFrame` at the boundary; `chat`→turn, `steer`→inline send, `interrupt`→interrupt. Egress serializes a `TurnFrame` (visible_text + engine-filled typed stance) into the payload: `chat_reply` on `direct(chatd)`, `turn` on `agent_status`. No done — orchestrator derives it. |
| `outbound_guard.py` | Buffer-complete-then-sanitize reply guard: strip internal markers/headers (unicode-normalization defense), redact credential-shaped strings. Unconditional on the reply path. |
| `config.py` | Typed `AgentConfig` from `agents/<name>/agent.yaml`; `*_ref` credential indirection (names, never literal secrets). |
| `persona_glue.py` | Compose the system prompt via `rtech_persona` (`memory_block=None` until Phase 4). |
| `runtime.py` | Wire supervisor + ingress onto a FleetBus (shared by the daemon, live test, and skeleton runner). |

## Dependencies

- `claude-agent-sdk` (PyPI) — the v1 engine.
- `nats-py` — NATS client (via `fleet-nats`).
- `fleet-nats`, `fleet-subscriber` — the fleet-bus wire contract and the **pure**
  subscriber pieces (`mapping` / `role` / `completion`). Never import
  `fleet_subscriber.hermes` / `fleet_subscriber.adapter` (Hermes-gateway seam).
- `rtech-persona` — persona load + compose.

`fleet-nats` and `fleet-subscriber` are wired as **local path sources** in the
root `[tool.uv.sources]` for dev/CI on this checkout. **Deploys pin these as git
deps** (fleet-bus repo) rather than a colocated path — swap the path source for a
`{ git = "…fleet-bus…", subdirectory = "packages/fleet-nats" }` source at deploy
time.

## Skeleton proof

```bash
ENGINE=fake   bash scripts/skeleton-roundtrip.sh   # scripted engine (CI)
ENGINE=claude bash scripts/skeleton-roundtrip.sh   # real Claude Agent SDK
```

Publishes a `chat` envelope, asserts the reply self-identifies per the test SOUL
(`agents/_skeleton/SOUL.md`), injects a mid-turn `steer`, asserts acknowledgement,
and writes `.evidence/skeleton-<ts>.json`. Uses live NATS when reachable, else an
in-process bus. Exits 0 only on full proof.
