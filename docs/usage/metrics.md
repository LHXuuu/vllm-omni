# Production Metrics

vLLM-Omni exposes Prometheus metrics via the `/metrics` endpoint on the
OpenAI-compatible API server. The metrics fall into three categories depending
on the pipeline type.

```bash
vllm-omni serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000
curl http://localhost:8000/metrics
```

## Metric Namespaces

| Prefix | Source | Present when |
|--------|--------|--------------|
| `vllm:omni_` | vLLM-Omni orchestrato / diffusion stages | Always / Pipeline includes a diffusion stage |
| `vllm:` | Upstream vLLM engine | Pipeline includes an LLM (AR) stage |
| `http_` / `process_` | Uvicorn / Python runtime | Always |

## Pipeline-Level Metrics (`vllm:omni_`)

These metrics are defined in `vllm_omni/metrics/prometheus.py` and track
request lifecycle across the full multi-stage pipeline.

### Request Tracking

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_num_requests_running` | Gauge | `model_name` | Requests currently running across all pipeline stages |
| `vllm:omni_num_requests_waiting` | Gauge | `model_name` | Requests waiting to be scheduled |
| `vllm:omni_num_requests_success` | Counter | `model_name` | Requests that completed without error |
| `vllm:omni_num_requests_fail` | Counter | `model_name` | Requests that returned an error |

### Latency

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_e2e_request_latency_seconds` | Histogram | `model_name` | End-to-end request latency in seconds |
| `vllm:omni_request_queue_time_seconds` | Histogram | `model_name` | Time spent waiting in the request queue |

## Diffusion Engine Metrics (`vllm:omni_`)

These histograms are populated only when the pipeline includes a diffusion
stage (e.g. image or video generation models).

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_diffusion_preprocess_time_ms` | Histogram | `model_name`, `engine` | Input preprocessing time per request |
| `vllm:omni_diffusion_exec_time_ms` | Histogram | `model_name`, `engine` | DiT forward pass execution time per request |
| `vllm:omni_diffusion_postprocess_time_ms` | Histogram | `model_name`, `engine` | Output postprocessing time (VAE decode) per request |
| `vllm:omni_diffusion_step_time_ms` | Histogram | `model_name`, `engine` | Total diffusion step time per request |

## vLLM Engine Metrics (`vllm:`)

When the pipeline includes an LLM stage, the upstream vLLM engine exposes its
full set of metrics under the `vllm:` prefix. These are registered by
`vllm.v1.metrics.loggers.PrometheusStatLogger` and cover scheduler state,
token throughput, cache utilization, and request latencies.

For a full overview of vLLM metrics, consult [the vLLM docs](https://github.com/vllm-project/vllm/blob/main/docs/usage/metrics.md)

## Metric Availability by Pipeline Type

| Metric group | Multi-stage LLM (Qwen3-Omni) | Diffusion-only (Z-Image-Turbo) |
|---|---|---|
| `vllm:omni_` request tracking | Yes | Yes |
| `vllm:omni_` latency | Yes | Yes |
| `vllm:omni_` KV cache | Yes | No |
| `vllm:omni_` diffusion timing | Only if pipeline has a diffusion stage | Yes |
| `vllm:` engine metrics | Yes | No |
| `vllm:` MFU metrics | With `--enable-mfu-metrics` | No |

## Naming Convention

vLLM-Omni pipeline metrics use the `vllm:omni_` prefix to distinguish
them from upstream per-engine `vllm:` metrics. The upstream
`unregister_vllm_metrics()` function is monkey-patched to a no-op (see
`vllm_omni/patch.py`) so that these metrics are not destroyed during
engine initialization.
