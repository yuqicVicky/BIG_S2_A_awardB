"""Append-only progress heartbeat for the modeling engine.

Env-gated and dataset-agnostic. When ``AWARDB_HEARTBEAT_PATH`` is unset this is a
no-op, so default behaviour is completely unchanged. When it is set, :func:`emit`
appends one JSON line per training milestone — a completed candidate, a tuned
family, a seed — that the ``modeling-watchdog`` agent reads *live* to keep a run
inside the time budget the orchestrator derived for it.

A logging failure must never propagate into training, so every write is guarded.
No policy lives here: thresholds and decisions belong to the watchdog agent; this
module only exposes progress.
"""

from __future__ import annotations

import json
import os
import time

_T0 = time.monotonic()


def emit(event: str, **fields) -> None:
    """Append one heartbeat line. No-op unless ``AWARDB_HEARTBEAT_PATH`` is set."""
    path = os.environ.get("AWARDB_HEARTBEAT_PATH")
    if not path:
        return
    try:
        line = {
            "ts": time.time(),
            "event": event,
            "elapsed_sec": round(time.monotonic() - _T0, 3),
            "role": os.environ.get("AWARDB_ROLE"),
            "run_id": os.environ.get("AWARDB_RUN_ID"),
        }
        line.update(fields)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
            fh.flush()
    except Exception:
        # Heartbeat logging is best-effort; never break a training run over it.
        pass
