# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import logging
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_INPUT_AUDIO_EVENTS = frozenset({"input_audio_buffer.append", "input.audio.append"})
_OUTPUT_AUDIO_EVENTS = frozenset({"response.audio.delta", "response.output_audio.delta"})
_SENSITIVE_ACTION_FIELDS = frozenset(
    {
        "audio",
        "content",
        "data",
        "delta",
        "error",
        "instructions",
        "message",
        "payload",
        "text",
        "tools",
        "transcript",
    }
)


def _audio_base64_chars(event: dict[str, Any]) -> int | None:
    event_type = event.get("type")
    if event_type in _INPUT_AUDIO_EVENTS:
        candidates = ("audio", "data")
    elif event_type in _OUTPUT_AUDIO_EVENTS:
        candidates = ("delta", "audio")
    else:
        return None
    for key in candidates:
        value = event.get(key)
        if isinstance(value, str) and value:
            return len(value)
    return None


def summarize_realtime_event(
    event: dict[str, Any],
    *,
    session_id: str | None = None,
) -> dict[str, object]:
    """Return a bounded event summary without text or Base64 payloads."""
    event_type = event.get("type")
    summary: dict[str, object] = {"event": event_type if isinstance(event_type, str) else "unknown"}
    for key in (
        "session_id",
        "event_id",
        "item_id",
        "response_id",
        "request_id",
        "turn_id",
        "epoch",
        "code",
        "status",
        "audio_start_ms",
        "audio_end_ms",
        "sample_rate_hz",
        "format",
        "finish_reason",
    ):
        value = event.get(key)
        if isinstance(value, str | int | float | bool):
            summary[key] = value

    session = event.get("session")
    if isinstance(session, dict):
        nested_session_id = session.get("id")
        if isinstance(nested_session_id, str):
            summary["session_id"] = nested_session_id
        model = session.get("model")
        if isinstance(model, str):
            summary["model"] = model
        turn_detection_present = "turn_detection" in session
        turn_detection = session.get("turn_detection")
        audio = session.get("audio")
        audio_input = audio.get("input") if isinstance(audio, dict) else None
        if not turn_detection_present and isinstance(audio_input, dict) and "turn_detection" in audio_input:
            turn_detection_present = True
            turn_detection = audio_input.get("turn_detection")
        if turn_detection_present:
            if isinstance(turn_detection, dict) and isinstance(turn_detection.get("type"), str):
                summary["turn_detection"] = turn_detection["type"]
            elif turn_detection is None:
                summary["turn_detection"] = None

    response = event.get("response")
    if isinstance(response, dict):
        response_id = response.get("id")
        if isinstance(response_id, str):
            summary["response_id"] = response_id
        response_status = response.get("status")
        if isinstance(response_status, str):
            summary["status"] = response_status

    audio_chars = _audio_base64_chars(event)
    if audio_chars is not None:
        summary["audio_base64_chars"] = audio_chars
        summary["audio_bytes_estimate"] = audio_chars * 3 // 4

    for key in ("text", "delta", "transcript"):
        value = event.get(key)
        if isinstance(value, str) and value and not (key == "delta" and audio_chars is not None):
            summary[f"{key}_chars"] = len(value)

    error = event.get("error")
    if isinstance(error, str) and error:
        summary["error_chars"] = len(error)
    elif isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            summary["error_chars"] = len(message)
        error_type = error.get("type")
        if isinstance(error_type, str):
            summary["error_type"] = error_type
        error_code = error.get("code")
        if isinstance(error_code, str):
            summary["code"] = error_code
    if session_id:
        # The bound Session context is authoritative. Event fields are
        # untrusted wire data and must not redirect trace correlation.
        summary["session_id"] = session_id
    return summary


def trace_realtime_event(
    component: str,
    direction: str,
    event: dict[str, Any],
    *,
    session_id: str | None = None,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    payload = {
        "component": component,
        "direction": direction,
        **summarize_realtime_event(event, session_id=session_id),
    }
    logger.debug("[RealtimeTrace] %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))


def trace_realtime_action(
    component: str,
    action: str,
    *,
    session_id: str | None = None,
    **fields: object,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    payload: dict[str, object] = {"component": component, "action": action}
    if session_id:
        payload["session_id"] = session_id
    for key, value in fields.items():
        if key not in _SENSITIVE_ACTION_FIELDS:
            payload[key] = value
        elif isinstance(value, str):
            payload[f"{key}_chars"] = len(value)
        elif isinstance(value, bytes | bytearray | memoryview):
            payload[f"{key}_bytes"] = len(value)
        else:
            payload[f"{key}_present"] = value is not None
    logger.debug("[RealtimeTrace] %s", json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def log_realtime_trace_configuration() -> None:
    trace_realtime_action(
        "serving",
        "trace_enabled",
        payloads=False,
    )


__all__ = [
    "log_realtime_trace_configuration",
    "summarize_realtime_event",
    "trace_realtime_action",
    "trace_realtime_event",
]
