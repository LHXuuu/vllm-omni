from __future__ import annotations

import json
from unittest.mock import Mock

from vllm_omni.experimental.fullduplex.openai import realtime_trace


def test_realtime_event_summary_redacts_audio_and_text() -> None:
    summary = realtime_trace.summarize_realtime_event(
        {
            "type": "input_audio_buffer.append",
            "event_id": "event-1",
            "audio": "sensitive-base64-audio",
            "text": "sensitive-text",
            "sample_rate_hz": 16_000,
        },
        session_id="session-1",
    )

    assert summary == {
        "event": "input_audio_buffer.append",
        "session_id": "session-1",
        "event_id": "event-1",
        "sample_rate_hz": 16_000,
        "audio_base64_chars": len("sensitive-base64-audio"),
        "audio_bytes_estimate": len("sensitive-base64-audio") * 3 // 4,
        "text_chars": len("sensitive-text"),
    }
    assert "sensitive" not in json.dumps(summary)


def test_realtime_trace_requires_debug_logging(monkeypatch) -> None:
    debug = Mock()
    debug_enabled = Mock(return_value=False)
    monkeypatch.setattr(realtime_trace.logger, "debug", debug)
    monkeypatch.setattr(realtime_trace.logger, "isEnabledFor", debug_enabled)

    event = {"type": "response.audio.delta", "delta": "sensitive-base64-audio"}
    realtime_trace.trace_realtime_event("websocket", "server_event", event)
    debug.assert_not_called()

    debug_enabled.return_value = True
    realtime_trace.trace_realtime_event("websocket", "server_event", event)
    debug.assert_called_once()
    rendered = debug.call_args.args[1]
    assert "response.audio.delta" in rendered
    assert "sensitive-base64-audio" not in rendered


def test_realtime_text_delta_is_counted_as_text() -> None:
    summary = realtime_trace.summarize_realtime_event({"type": "response.text.delta", "delta": "sensitive-text"})

    assert summary == {
        "event": "response.text.delta",
        "delta_chars": len("sensitive-text"),
    }


def test_realtime_session_summary_includes_turn_detection_without_instructions() -> None:
    summary = realtime_trace.summarize_realtime_event(
        {
            "type": "session.update",
            "session": {
                "model": "model-a",
                "instructions": "sensitive-instructions",
                "audio": {
                    "input": {
                        "turn_detection": {"type": "server_vad", "threshold": 0.5},
                    }
                },
            },
        }
    )

    assert summary == {
        "event": "session.update",
        "model": "model-a",
        "turn_detection": "server_vad",
    }


def test_realtime_summary_redacts_errors_and_preserves_bound_session() -> None:
    summary = realtime_trace.summarize_realtime_event(
        {
            "type": "error",
            "session_id": "untrusted-session",
            "error": {
                "type": "invalid_request_error",
                "code": "bad_event",
                "message": "sensitive-input-value",
            },
        },
        session_id="bound-session",
    )

    assert summary == {
        "event": "error",
        "session_id": "bound-session",
        "code": "bad_event",
        "error_chars": len("sensitive-input-value"),
        "error_type": "invalid_request_error",
    }
    assert "sensitive-input-value" not in json.dumps(summary)


def test_realtime_summary_does_not_invent_turn_detection_null() -> None:
    summary = realtime_trace.summarize_realtime_event(
        {"type": "session.update", "session": {"audio": {"input": {"format": "pcm16"}}}}
    )

    assert summary == {"event": "session.update"}
