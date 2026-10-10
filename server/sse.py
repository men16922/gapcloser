"""Server-Sent Events for a job that appends events under a threading.Condition (console runs, Studio analyses)."""

from __future__ import annotations

import json

from fastapi.responses import StreamingResponse

HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def event_stream(job) -> StreamingResponse:
    """Replay `job.events` from the start, then follow new ones until `job.done`. `job` has .events, .done, .cond."""

    def stream():
        i = 0
        while True:
            with job.cond:
                while i >= len(job.events) and not job.done:
                    if not job.cond.wait(timeout=15):
                        break
                batch, done = job.events[i:], job.done
            if not batch and not done:
                yield ": keep-alive\n\n"
                continue
            for e in batch:
                yield f"data: {json.dumps(e, default=str)}\n\n"
            i += len(batch)
            if done and i >= len(job.events):
                return

    return StreamingResponse(stream(), media_type="text/event-stream", headers=HEADERS)
