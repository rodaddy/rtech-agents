# rtech-chatd

Chat ingress/egress daemon behind a `ChannelAdapter` protocol. v1 implements the
Discord behavioral contract (mention-gate, `allow_bots >= mentions`, auto-thread,
fence-aware chunking, typing loop, allowlists). WhatsApp is the next transport
under the same gate layer.

chatd is a dumb edge on the typed bus (`rtech_frames`):

- gated inbound human input mints a `TaskFrame` (`origin_edge="discord"`,
  `scope="single"`) published on `rtech.task.new`;
- it subscribes each task's `rtech.task.<id>.turn` / `.status` subjects and
  renders `visible_text` / a minimal status indicator back into the originating
  thread;
- every outbound render runs the whole buffered text through the egress guard
  before send — streaming never bypasses the guard;
- stance, refs, done state, participants, and control signals are not interpreted
  by the connector. Done is orchestrator-derived from the frame graph.

## Wire (`edge.py` + `bus.py` + `subjects.py`)

`DiscordEdge` ties the `ChannelAdapter` to a `FrameBus` (FleetBus over raw frame
bytes). Subjects mirror the ratified contract: `rtech.task.new` ingress plus the
per-task `turn`/`status`/`steer` fan-out.

## Credentials + home channel

- The bot token is resolved via `*_ref` indirection only (`config.resolve_token`,
  `env:RTECH_CHATD_BOT_TOKEN`) — never a literal. Only `env:VAR` refs are accepted
  at runtime (`config.validate_token_ref` fails closed on anything else). The
  secret VALUE may be sourced from Vaultwarden by the operator, but it is injected
  into the env var the ref names — configure `env:VAR`, never a `vaultwarden:` ref.
- Each agent has a Hermes-shaped home channel (`HomeChannel{chat_id, name,
  thread_id}`, a bare channel **id**, not a ref) read from
  `DISCORD_HOME_CHANNEL[_NAME|_THREAD_ID]`. It is where cron results and
  cross-platform messages land. A missing/invalid home channel fails loudly in
  **logs only** at startup; the `📬` advisory is never posted as inbound noise —
  it surfaces to Discord only on a genuine failed `deliver_to_home` attempt.

## Live canary

`scripts/discord_canary.py` proves mention → auto-thread → reply. It is gated on
`DISCORD_LIVE=1` + a resolvable token ref; it SKIPs (exit 0) otherwise and is run
on the deploy target, not in CI.

Phase 5 of the governing spec.
