# rtech-webchat

Dumb FastAPI web-chat edge for the bus-first POC.

It mints typed `rtech-frames` frames, encodes them with `rtech_frames.codec`, and
publishes/subscribes on NATS. It does not infer completion, parse stance, manage
participants, or carry connector-local control state.

The browser UI is one self-contained HTML page with inline CSS/JS and no build
step. It keeps a task list for tasks minted by this edge, renders observed
`TurnFrame` and `StatusFrame` payloads in arrival order, and marks result-subject
arrival as a plain `result` event. The edge does not compute done state.

## Websocket Choice

The UI uses one `/ws` socket and sends `{"type":"subscribe","task_id":"..."}`
commands for each task. The server then subscribes that socket to the task's
turn, status, and result subjects. This keeps one browser connection per
operator page while preserving bus-native per-task subjects; the edge only routes
observed frames to the browser and does not coordinate task state.

## Subject Scheme

The ratified subject scheme lives in `rtech_webchat.subjects`:

| Purpose | Subject |
| --- | --- |
| Task ingress | `rtech.task.new` |
| Per-task turns | `rtech.task.<task_id>.turn` |
| Per-task status | `rtech.task.<task_id>.status` |
| Per-task steer | `rtech.task.<task_id>.steer` |
| Per-task result | `rtech.task.<task_id>.result` |
| Agent dispatch | `rtech.agent.<agent_id>.task` |

## Runtime

`RTECH_WEBCHAT_NATS_URL` controls the NATS URL and defaults to
`nats://127.0.0.1:4222`. The HTTP bind host defaults to `127.0.0.1`.

```bash
uv run rtech-webchat
```
