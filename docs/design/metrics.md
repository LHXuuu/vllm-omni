# Prometheus Metrics Design

This document describes how vLLM-Omni exposes Prometheus metrics for
multi-stage pipelines, the constraints that shaped the design, and how
the pipeline-level metrics coexist with upstream vLLM per-engine
metrics.

## Objectives

- Expose pipeline-level request and latency metrics that span the full
  multi-stage execution (orchestrator scope).
- Preserve all upstream vLLM per-engine metrics (`vllm:*`) for stages
  backed by an AR LLM engine.
- Expose per-stage diffusion timing breakdowns for pipelines that
  include a diffusion engine.
- Keep the metrics collection overhead low enough that it does not
  regress TTFA or throughput.

## Background

### Upstream vLLM Metrics

Upstream vLLM defines 44 Prometheus metrics under the `vllm:` prefix.
These are registered by `PrometheusStatLogger` and cover engine-level
state: KV cache usage, running/waiting request counts, token
throughput, TTFT, inter-token latency, e2e latency, and so on. They
are served via the `/metrics` HTTP endpoint provided by
`prometheus_fastapi_instrumentator` and the default
`prometheus_client` WSGI handler.

vLLM's `unregister_vllm_metrics()` function strips every
`prometheus_client` collector whose `_name` attribute contains the
substring `"vllm"`. This runs during engine initialization to clean up
stale collectors from prior instantiations within the same process.

### The Problem

vLLM-Omni runs multiple engine instances (stages) within a single
process, coordinated by an Orchestrator. The pipeline needs its own
metrics — aggregate request counts, end-to-end latency across all
stages, and diffusion timing breakdowns — that do not exist in upstream
vLLM. All pipeline-level metrics use the `vllm:omni_` prefix to
distinguish them from upstream per-engine metrics. The
`unregister_vllm_metrics()` function is monkey-patched to a no-op at
import time (see `vllm_omni/patch.py`) so that these metrics are not
destroyed during engine initialization.

Upstream per-engine metrics retain the `vllm:` prefix and are
registered by a `PrometheusStatLogger` instance that the Orchestrator
creates and feeds directly.

## Architecture

### Component Overview

```
                       +-----------------------+
                       |    API Server (FastAPI)|
                       |   GET /metrics         |
                       +----------+------------+
                                  |
                     prometheus_client default handler
                                  |
                    +-------------+-------------+
                    |                           |
          vllm:omni_* collectors      vllm:* collectors
                    |                           |
        +-----------+-----------+      +--------+---------+
        | OmniPrometheusMetrics |      | PrometheusStatLogger |
        +-----------+-----------+      +--------+---------+
                    |                           |
               OmniBase                   Orchestrator
            (request lifecycle,       (feeds SchedulerStats
             diffusion timing)        + IterationStats
                                       per engine step)
```

### Data Flow

There are two independent paths for metric collection.

**Path 1: Pipeline-level metrics (`vllm:omni_*`)**

`OmniPrometheusMetrics` registers Gauge, Counter, and Histogram
collectors at init time. It is instantiated once per entrypoint,
labeled with the model name. The entrypoint calls its methods as
requests progress:

- `set_running(n)` / `set_waiting(n)` — updated after each request
  completes. The running count comes from `OmniRequestCounter`, a
  simple counter incremented/decremented by the Orchestrator as it
  tracks requests. Waiting is derived as `total - running`.

- `request_succeeded(e2e_seconds, queue_seconds)` — recorded when a
  request finishes at the final stage.

- `request_failed()` — recorded when a request errors.

- `observe_diffusion_metrics(stage_id, metrics)` — recorded when a
  diffusion stage finishes. The metrics dict contains timing
  breakdowns (preprocess, exec, postprocess, total step time)
  accumulated from engine output.

**Path 2: Per-engine metrics (`vllm:*`)**

The Orchestrator instantiates upstream vLLM's `PrometheusStatLogger`
and feeds it scheduler stats and iteration stats after processing
each batch of engine outputs. This populates the standard vLLM
metrics (TTFT, token throughput, cache usage, etc.) using the same
code path as standalone vLLM. For diffusion-only pipelines that have
no AR engine, `SchedulerStats` is never produced and `vllm:*` metrics
are absent.

### Shared State Between Threads

The Orchestrator runs in a background thread. The API server
(OmniBase) runs in the asyncio event loop thread.
`OmniRequestCounter` bridges them — a plain Python object with an
`int` field. The Orchestrator increments/decrements it; the
entrypoint reads it for gauge updates. No lock is needed because the
counter is advisory (a stale read by one Prometheus scrape interval
is acceptable). It is created by `AsyncOmniEngine.__init__()` and
passed to the Orchestrator at construction time.

### Metric Registration and Lifecycle

All `vllm:omni_*` collectors are registered once when
`OmniPrometheusMetrics.__init__()` runs. Per-stage labels
(`model_name`, `engine`) are bound lazily on first observation to
avoid registering labels for stages that never produce data (e.g., a
diffusion pipeline has no AR stage stats).

The `prometheus_client` default registry holds all collectors.
FastAPI's `/metrics` endpoint serves the default registry, so both
`vllm:omni_*` and `vllm:*` metrics appear in the same scrape
response alongside `http_*` and `process_*` metrics from the
instrumentator and the Python client runtime.

## Throttling: `make_stats()` Override

Upstream vLLM's `Scheduler.make_stats()` runs on every AR generation step,
returning a SchedulerStats object for the orchestrator.
Under vLLM's architecture, this is fine. But since vLLM-Omni requires that the
object be serialized and transferred over ZMQ, receiving a SchedulerStats object on
every step can introduce unacceptable overhead to the system.

`OmniSchedulerMixin.make_stats()` (in
`vllm_omni/core/sched/omni_scheduler_mixin.py`) throttles stats
emission to at most once per second. Between intervals it returns
`None`, which the engine core skips serializing. This keeps gauges
fresh enough for Prometheus scrapes (typically 15-30s intervals) while
eliminating the per-step overhead.

## Metric Definitions

### Pipeline-Level

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_num_requests_running` | Gauge | `model_name` | Requests currently executing across all stages |
| `vllm:omni_num_requests_waiting` | Gauge | `model_name` | Requests queued but not yet scheduled |
| `vllm:omni_num_requests_success` | Counter | `model_name` | Requests completed without error |
| `vllm:omni_num_requests_fail` | Counter | `model_name` | Requests that returned an error |
| `vllm:omni_e2e_request_latency_seconds` | Histogram | `model_name` | End-to-end request latency across all stages |
| `vllm:omni_request_queue_time_seconds` | Histogram | `model_name` | Time spent waiting in the request queue |

### Diffusion Stage-Level

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_diffusion_preprocess_time_ms` | Histogram | `model_name`, `engine` | Diffusion input preprocessing time |
| `vllm:omni_diffusion_exec_time_ms` | Histogram | `model_name`, `engine` | Diffusion model forward pass time |
| `vllm:omni_diffusion_postprocess_time_ms` | Histogram | `model_name`, `engine` | Diffusion output postprocessing time |
| `vllm:omni_diffusion_step_time_ms` | Histogram | `model_name`, `engine` | Total diffusion step time |

### LLM Stage-Level

Reference [vLLM docs](https://github.com/vllm-project/vllm/blob/main/docs/usage/metrics.md)

Note that metrics that depend upon features that are not supported in vLLM-Omni (e.g. speculative decoding, LoRA) will not be available as well.

## Logging vs. Prometheus

`OrchestratorAggregator` (in `vllm_omni/metrics/stats.py`) is the
logging-oriented metrics path. It collects detailed per-request,
per-stage, and per-transfer statistics and prints formatted tables to
the `INFO` log. This is designed for development and debugging —
individual request traces, transfer bandwidth, inter-stage timing.

`OmniPrometheusMetrics` is the Prometheus-oriented path. It records
aggregate counters, gauges, and histograms suitable for time-series
monitoring and alerting. The two paths are independent; both can run
simultaneously.

The separation follows upstream vLLM's pattern of `LoggingStatLogger`
vs. `PrometheusStatLogger` — same underlying data, different
consumption models.
