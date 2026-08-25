# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from vllm_omni.metrics import definitions as defs

_active_sessions = Gauge(
    defs.REALTIME_VAD_ACTIVE_SESSIONS,
    "Number of active Realtime sessions using server-side VAD.",
    labelnames=list(defs.REALTIME_VAD_LABELS),
)
_inference_latency = Histogram(
    defs.REALTIME_VAD_INFERENCE_LATENCY_S,
    "Server VAD inference time for one appended audio batch, in seconds.",
    labelnames=list(defs.REALTIME_VAD_LABELS),
    buckets=defs.SECONDS_FAST_BUCKETS,
)
_endpoint_delay = Histogram(
    defs.REALTIME_VAD_ENDPOINT_DELAY_S,
    "Detected silence between the last speech frame and the committed endpoint, in seconds.",
    labelnames=list(defs.REALTIME_VAD_LABELS),
    buckets=defs.SECONDS_FAST_BUCKETS,
)
_errors = Counter(
    defs.REALTIME_VAD_ERRORS,
    "Server VAD errors and input-overflow rejections by reason.",
    labelnames=list(defs.REALTIME_VAD_ERROR_LABELS),
)


class RealtimeVADMetrics:
    def session_started(self, model_name: str) -> None:
        _active_sessions.labels(model_name=model_name).inc()

    def session_finished(self, model_name: str) -> None:
        _active_sessions.labels(model_name=model_name).dec()

    def observe_inference(self, model_name: str, latency_ms: float) -> None:
        _inference_latency.labels(model_name=model_name).observe(max(0.0, latency_ms) / 1000)

    def observe_endpoint_delay(self, model_name: str, delay_ms: int) -> None:
        _endpoint_delay.labels(model_name=model_name).observe(max(0, delay_ms) / 1000)

    def error(self, model_name: str, reason: str) -> None:
        _errors.labels(model_name=model_name, reason=reason).inc()
