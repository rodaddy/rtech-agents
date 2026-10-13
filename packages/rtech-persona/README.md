# rtech-persona

Prompt composition library. Loads `SOUL.md` (identity), `USER.md` (profile), and
`KERNEL.md` (rules) from an agent home, runs a Hermes-derived content scan and a
context-budget-scaled length cap, and composes the ordered system-prompt slots:
identity → kernel → user profile → memory block → platform hint → timestamp/model
line.

See `src/rtech_persona/loader.py` and `src/rtech_persona/composer.py`.
