# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from vllm.transformers_utils.repo_utils import try_get_local_file

SILERO_VAD_REPO_ID = "istupakov/silero-vad-onnx"
SILERO_VAD_REVISION = "8b14476858ef240c50b3884bb38cc67290c1cc70"
SILERO_VAD_FILENAME = "silero_vad.onnx"
SILERO_VAD_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

_SERVER_VAD_FIELDS = {
    "type",
    "threshold",
    "prefix_padding_ms",
    "silence_duration_ms",
    "create_response",
    "interrupt_response",
}


@dataclass(frozen=True, slots=True)
class ServerVADConfig:
    """Validated OpenAI-compatible ``server_vad`` session configuration."""

    type: Literal["server_vad"] = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    create_response: bool = True
    interrupt_response: bool = False

    @classmethod
    def from_value(cls, value: object) -> ServerVADConfig:
        if not isinstance(value, dict):
            raise ValueError("turn_detection must be null or an object")
        unknown = sorted(set(value) - _SERVER_VAD_FIELDS)
        if unknown:
            raise ValueError(f"Unknown server_vad field(s): {', '.join(unknown)}")
        vad_type = value.get("type")
        if vad_type != "server_vad":
            raise ValueError("turn_detection.type must be 'server_vad'")

        threshold = value.get("threshold", 0.5)
        if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0 <= threshold <= 1:
            raise ValueError("server_vad.threshold must be a number between 0 and 1")

        prefix_padding_ms = value.get("prefix_padding_ms", 300)
        if isinstance(prefix_padding_ms, bool) or not isinstance(prefix_padding_ms, int) or prefix_padding_ms < 0:
            raise ValueError("server_vad.prefix_padding_ms must be a non-negative integer")

        silence_duration_ms = value.get("silence_duration_ms", 500)
        if (
            isinstance(silence_duration_ms, bool)
            or not isinstance(silence_duration_ms, int)
            or silence_duration_ms <= 0
        ):
            raise ValueError("server_vad.silence_duration_ms must be a positive integer")

        create_response = value.get("create_response", True)
        if not isinstance(create_response, bool):
            raise ValueError("server_vad.create_response must be a boolean")

        interrupt_response = value.get("interrupt_response", False)
        if not isinstance(interrupt_response, bool):
            raise ValueError("server_vad.interrupt_response must be a boolean")
        if interrupt_response:
            raise ValueError("server_vad.interrupt_response=true is not supported")

        return cls(
            type=vad_type,
            threshold=float(threshold),
            prefix_padding_ms=prefix_padding_ms,
            silence_duration_ms=silence_duration_ms,
            create_response=create_response,
            interrupt_response=False,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
            "create_response": self.create_response,
            "interrupt_response": self.interrupt_response,
        }


class SpeechDetectorBackend(Protocol):
    """Detector contract; per-stream state is owned by the pipeline."""

    frame_samples: int

    def new_state(self) -> object: ...

    def infer(self, frame: np.ndarray, state: object) -> tuple[float, object]: ...


class SpeechDetectorBackendProvider(Protocol):
    def get(self) -> SpeechDetectorBackend: ...


@dataclass(frozen=True, slots=True)
class SpeechEndpointDecision:
    speech_started: bool = False
    speech_stopped: bool = False
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None
    endpoint_delay_ms: int | None = None


class SpeechEndpointPolicy(Protocol):
    @property
    def speech_active(self) -> bool: ...

    def update(
        self,
        probability: float,
        *,
        frame_start_sample: int,
        frame_samples: int,
    ) -> SpeechEndpointDecision: ...

    def reset(self) -> None: ...


class _StreamingPCM16MonoResampler:
    """Polyphase resampler for a continuous chunked mono PCM16 stream.

    The symmetric Kaiser-windowed sinc filter matches the default filter used
    by ``scipy.signal.resample_poly`` without adding SciPy as a serving
    dependency. ``push`` emits only samples whose right-hand filter support is
    available, so output is invariant to arbitrary WebSocket chunking. The
    resulting lookahead is 0.625 ms for 24 kHz to 16 kHz conversion.

    ``flush`` supplies the right-edge zero padding required by a finite input
    stream. A Realtime session remains continuous across speech turns and
    therefore does not flush at VAD endpoints; ``reset`` intentionally drops
    the buffered tail when the input buffer is cleared.
    """

    _half_filter_width = 10
    _kaiser_beta = 5.0

    def __init__(self, *, source_rate_hz: int, target_rate_hz: int = 16_000) -> None:
        if isinstance(source_rate_hz, bool) or not isinstance(source_rate_hz, int) or source_rate_hz <= 0:
            raise ValueError("source_rate_hz must be a positive integer")
        if isinstance(target_rate_hz, bool) or not isinstance(target_rate_hz, int) or target_rate_hz <= 0:
            raise ValueError("target_rate_hz must be a positive integer")
        self.source_rate_hz = source_rate_hz
        self.target_rate_hz = target_rate_hz
        rate_gcd = math.gcd(source_rate_hz, target_rate_hz)
        self._up = target_rate_hz // rate_gcd
        self._down = source_rate_hz // rate_gcd
        self._half_len, self._phase_kernels = self._design_polyphase_filter()
        self._history_samples = self._phase_kernels.shape[1] - 1
        self._history = np.zeros(self._history_samples, dtype=np.float32)
        self._input_samples = 0
        self._output_samples = 0
        self._flushed = False

    def _design_polyphase_filter(self) -> tuple[int, np.ndarray]:
        if self._up == self._down:
            return 0, np.ones((1, 1), dtype=np.float32)

        max_rate = max(self._up, self._down)
        half_len = self._half_filter_width * max_rate
        offsets = np.arange(-half_len, half_len + 1, dtype=np.float64)
        cutoff = 1.0 / max_rate
        taps = cutoff * np.sinc(cutoff * offsets)
        taps *= np.kaiser(taps.size, self._kaiser_beta)
        taps /= np.sum(taps)
        taps *= self._up
        taps = np.ascontiguousarray(taps, dtype=np.float32)

        phases = [taps[phase :: self._up] for phase in range(self._up)]
        phase_width = max(phase.size for phase in phases)
        kernels = np.zeros((self._up, phase_width), dtype=np.float32)
        for phase_index, phase in enumerate(phases):
            kernels[phase_index, -phase.size :] = phase[::-1]
        return half_len, kernels

    @property
    def pending_samples(self) -> int:
        total_output_samples = self._ceil_div(self._input_samples * self._up, self._down)
        return max(0, total_output_samples - self._output_samples)

    @property
    def scratch_bytes(self) -> int:
        return self.pending_samples * np.dtype(np.float32).itemsize

    @staticmethod
    def _ceil_div(numerator: int, denominator: int) -> int:
        return -(-numerator // denominator)

    def _render_outputs(
        self,
        combined: np.ndarray,
        *,
        first_output: int,
        output_count: int,
        input_start: int,
    ) -> np.ndarray:
        if output_count <= 0:
            return np.empty(0, dtype=np.float32)

        output_indexes = np.arange(first_output, first_output + output_count, dtype=np.int64)
        filter_indexes = output_indexes * self._down + self._half_len
        source_indexes = filter_indexes // self._up
        phases = filter_indexes % self._up
        windows = np.lib.stride_tricks.sliding_window_view(
            combined,
            self._phase_kernels.shape[1],
        )
        selected_windows = windows[source_indexes - input_start]
        return np.sum(
            selected_windows * self._phase_kernels[phases],
            axis=1,
            dtype=np.float32,
        )

    def push(self, samples: np.ndarray) -> np.ndarray:
        if self._flushed:
            raise RuntimeError("cannot push audio after the resampler has been flushed; call reset first")

        pcm16 = np.ascontiguousarray(samples, dtype="<i2").reshape(-1)
        if pcm16.size == 0:
            return np.empty(0, dtype=np.float32)

        source = pcm16.astype(np.float32) * np.float32(1.0 / 32768.0)
        combined = np.concatenate((self._history, source))
        next_input_samples = self._input_samples + source.size
        stable_output_samples = max(
            0,
            self._ceil_div(next_input_samples * self._up - self._half_len, self._down),
        )
        output = self._render_outputs(
            combined,
            first_output=self._output_samples,
            output_count=stable_output_samples - self._output_samples,
            input_start=self._input_samples,
        )
        if self._history_samples:
            self._history = np.ascontiguousarray(combined[-self._history_samples :], dtype=np.float32)
        self._input_samples = next_input_samples
        self._output_samples = stable_output_samples
        return output

    def flush(self) -> np.ndarray:
        if self._flushed:
            return np.empty(0, dtype=np.float32)

        total_output_samples = self._ceil_div(self._input_samples * self._up, self._down)
        output_count = total_output_samples - self._output_samples
        if output_count:
            last_output = total_output_samples - 1
            last_source_index = (last_output * self._down + self._half_len) // self._up
            right_padding = max(0, last_source_index - self._input_samples + 1)
            combined = np.concatenate((self._history, np.zeros(right_padding, dtype=np.float32)))
            output = self._render_outputs(
                combined,
                first_output=self._output_samples,
                output_count=output_count,
                input_start=self._input_samples,
            )
        else:
            output = np.empty(0, dtype=np.float32)
        self._output_samples = total_output_samples
        self._flushed = True
        return output

    def reset(self) -> None:
        self._history = np.zeros(self._history_samples, dtype=np.float32)
        self._input_samples = 0
        self._output_samples = 0
        self._flushed = False


class ThresholdEndpointPolicy:
    """Apply threshold and trailing-silence endpoint rules."""

    def __init__(self, config: ServerVADConfig, *, sample_rate_hz: int) -> None:
        self.config = config
        self.sample_rate_hz = sample_rate_hz
        self._speech_active = False
        self._silence_samples = 0

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def update(
        self,
        probability: float,
        *,
        frame_start_sample: int,
        frame_samples: int,
    ) -> SpeechEndpointDecision:
        if probability >= self.config.threshold:
            self._silence_samples = 0
            if not self._speech_active:
                self._speech_active = True
                detected_start_ms = round(frame_start_sample * 1000 / self.sample_rate_hz)
                return SpeechEndpointDecision(
                    speech_started=True,
                    audio_start_ms=max(0, detected_start_ms - self.config.prefix_padding_ms),
                )
            return SpeechEndpointDecision()

        if not self._speech_active:
            return SpeechEndpointDecision()

        # Match Silero v6.2's streaming VAD hysteresis: activation uses the
        # configured threshold, while potential speech end starts only below
        # ``max(threshold - 0.15, 0.01)``. Once a silence candidate exists,
        # intermediate frames keep elapsed time moving but cannot themselves
        # close the turn.
        negative_threshold = max(self.config.threshold - 0.15, 0.01)
        below_negative_threshold = probability < negative_threshold
        if not below_negative_threshold and self._silence_samples == 0:
            return SpeechEndpointDecision()

        self._silence_samples += frame_samples
        if not below_negative_threshold:
            return SpeechEndpointDecision()
        silence_limit = max(1, round(self.config.silence_duration_ms * self.sample_rate_hz / 1000))
        if self._silence_samples < silence_limit:
            return SpeechEndpointDecision()

        # OpenAI defines ``audio_end_ms`` as the end of the audio sent to the
        # model, including the trailing silence used for endpoint detection.
        audio_end_sample = frame_start_sample + frame_samples
        endpoint_delay_ms = round(self._silence_samples * 1000 / self.sample_rate_hz)
        self.reset()
        return SpeechEndpointDecision(
            speech_stopped=True,
            audio_end_ms=max(0, round(audio_end_sample * 1000 / self.sample_rate_hz)),
            endpoint_delay_ms=endpoint_delay_ms,
        )

    def reset(self) -> None:
        self._speech_active = False
        self._silence_samples = 0


@dataclass(frozen=True, slots=True)
class ServerVADFrame:
    samples: np.ndarray
    decision: SpeechEndpointDecision


@dataclass(frozen=True, slots=True)
class ServerVADBatch:
    frames: tuple[ServerVADFrame, ...]
    inference_ms: float


class ServerVADPipeline:
    """Frame continuous 16 kHz audio and apply acoustic endpointing.

    The pipeline retains only input-resampling residuals, a partial frame,
    detector state, and timing counters. Commit-eligible prefix and utterance
    audio remain session-owned.
    """

    sample_rate_hz = 16_000

    def __init__(
        self,
        backend: SpeechDetectorBackend,
        config: ServerVADConfig,
        *,
        endpoint_policy: SpeechEndpointPolicy | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.endpoint_policy = endpoint_policy or ThresholdEndpointPolicy(
            config,
            sample_rate_hz=self.sample_rate_hz,
        )
        self._input_resampler: _StreamingPCM16MonoResampler | None = None
        self._source_sample_rate_hz: int | None = None
        self._scratch = np.empty(0, dtype=np.float32)
        self._detector_state = backend.new_state()
        self._processed_samples = 0

    @property
    def speech_active(self) -> bool:
        return self.endpoint_policy.speech_active

    @property
    def source_sample_rate_hz(self) -> int | None:
        return self._source_sample_rate_hz

    @property
    def scratch_bytes(self) -> int:
        resampler_bytes = self._input_resampler.scratch_bytes if self._input_resampler is not None else 0
        return int(self._scratch.nbytes) + resampler_bytes

    async def push_pcm16(
        self,
        samples: bytes | bytearray | memoryview | np.ndarray,
        *,
        source_sample_rate_hz: int,
    ) -> ServerVADBatch:
        """Normalize chunked mono PCM16 input and run endpoint detection."""
        if isinstance(source_sample_rate_hz, bool) or source_sample_rate_hz not in {16_000, 24_000}:
            raise ValueError("server_vad PCM16 input sample rate must be 16000 or 24000 Hz")

        if isinstance(samples, np.ndarray):
            if samples.ndim != 1 or samples.dtype.kind != "i" or samples.dtype.itemsize != 2:
                raise ValueError("server_vad PCM16 samples must be a one-dimensional int16 array")
            pcm16 = np.ascontiguousarray(samples, dtype="<i2")
        elif isinstance(samples, bytes | bytearray | memoryview):
            raw = bytes(samples)
            if len(raw) % np.dtype("<i2").itemsize:
                raise ValueError("server_vad PCM16 input contains an incomplete sample")
            pcm16 = np.frombuffer(raw, dtype="<i2")
        else:
            raise ValueError("server_vad PCM16 input must be bytes or an int16 array")

        if self._source_sample_rate_hz is not None and source_sample_rate_hz != self._source_sample_rate_hz:
            raise ValueError("server_vad input sample rate cannot change within a continuous audio stream")
        if pcm16.size and self._source_sample_rate_hz is None:
            input_resampler = (
                None
                if source_sample_rate_hz == self.sample_rate_hz
                else _StreamingPCM16MonoResampler(
                    source_rate_hz=source_sample_rate_hz,
                    target_rate_hz=self.sample_rate_hz,
                )
            )
            self._source_sample_rate_hz = source_sample_rate_hz
            self._input_resampler = input_resampler

        if self._input_resampler is not None:
            normalized = self._input_resampler.push(pcm16)
        else:
            normalized = np.ascontiguousarray(pcm16, dtype=np.float32) / np.float32(32768.0)
        return await self.push(normalized)

    async def push(self, samples: np.ndarray) -> ServerVADBatch:
        normalized = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        return await asyncio.to_thread(self._push_sync, normalized)

    def _push_sync(self, samples: np.ndarray) -> ServerVADBatch:
        started_at = time.perf_counter()
        if self._scratch.size:
            samples = np.concatenate((self._scratch, samples))
        frame_samples = int(self.backend.frame_samples)
        complete_samples = samples.size - samples.size % frame_samples
        self._scratch = np.ascontiguousarray(samples[complete_samples:], dtype=np.float32)

        results: list[ServerVADFrame] = []
        for offset in range(0, complete_samples, frame_samples):
            frame = np.ascontiguousarray(samples[offset : offset + frame_samples], dtype=np.float32)
            frame_start = self._processed_samples
            probability, self._detector_state = self.backend.infer(frame, self._detector_state)
            probability = min(1.0, max(0.0, float(probability)))
            decision = self.endpoint_policy.update(
                probability,
                frame_start_sample=frame_start,
                frame_samples=frame_samples,
            )

            self._processed_samples += frame_samples
            results.append(
                ServerVADFrame(
                    samples=frame,
                    decision=decision,
                )
            )
        return ServerVADBatch(
            frames=tuple(results),
            inference_ms=(time.perf_counter() - started_at) * 1000,
        )

    def reset(self, *, clear_timeline: bool = False) -> None:
        self._input_resampler = None
        self._source_sample_rate_hz = None
        self._scratch = np.empty(0, dtype=np.float32)
        self._detector_state = self.backend.new_state()
        self.endpoint_policy.reset()
        if clear_timeline:
            self._processed_samples = 0


@dataclass(frozen=True, slots=True)
class _SileroVADState:
    model_state: np.ndarray
    context: np.ndarray


class SileroVADBackend:
    """Shared ONNX Runtime Silero v6.2 detector running on CPU."""

    sample_rate_hz = 16_000
    frame_samples = 512
    context_samples = 64
    model_state_shape = (2, 1, 128)

    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - platform packaging supplies ORT.
            raise RuntimeError("server_vad requires ONNX Runtime") from exc

        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1

        self.model_path = Path(model_path)
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )
        self._inference_lock = threading.Lock()
        input_names = {item.name for item in self._session.get_inputs()}
        if "input" not in input_names or "sr" not in input_names or "state" not in input_names:
            raise RuntimeError(f"Unsupported Silero ONNX input contract: {sorted(input_names)}")
        self._warm_up()

    def _warm_up(self) -> None:
        self.infer(np.zeros(self.frame_samples, dtype=np.float32), self.new_state())

    def new_state(self) -> _SileroVADState:
        return _SileroVADState(
            model_state=np.zeros(self.model_state_shape, dtype=np.float32),
            context=np.zeros((1, self.context_samples), dtype=np.float32),
        )

    def infer(self, frame: np.ndarray, state: object) -> tuple[float, object]:
        if not isinstance(state, _SileroVADState):
            raise TypeError("Silero detector state must be created by SileroVADBackend.new_state()")
        model_state = np.ascontiguousarray(state.model_state, dtype=np.float32)
        context = np.ascontiguousarray(state.context, dtype=np.float32)
        if model_state.shape != self.model_state_shape:
            raise ValueError(f"Silero model state must have shape {self.model_state_shape}, got {model_state.shape}")
        expected_context_shape = (1, self.context_samples)
        if context.shape != expected_context_shape:
            raise ValueError(f"Silero context must have shape {expected_context_shape}, got {context.shape}")

        audio = np.ascontiguousarray(frame, dtype=np.float32).reshape(1, -1)
        if audio.shape[1] != self.frame_samples:
            raise ValueError(
                f"Silero detector frame must contain exactly {self.frame_samples} samples, got {audio.shape[1]}"
            )
        model_input = np.concatenate((context, audio), axis=1)
        inputs = {
            "input": model_input,
            "state": model_state,
            "sr": np.asarray(self.sample_rate_hz, dtype=np.int64),
        }
        with self._inference_lock:
            output = self._session.run(None, inputs)
        if len(output) < 2:
            raise RuntimeError("Silero ONNX model did not return probability and model state")
        probability = float(np.asarray(output[0]).reshape(-1)[0])
        next_state = _SileroVADState(
            model_state=np.ascontiguousarray(output[1], dtype=np.float32),
            context=np.ascontiguousarray(model_input[:, -self.context_samples :], dtype=np.float32),
        )
        return probability, next_state


class SileroVADBackendProvider:
    """Resolve, verify, and load one Silero model instance per process."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        repo_id: str = SILERO_VAD_REPO_ID,
        revision: str = SILERO_VAD_REVISION,
        filename: str = SILERO_VAD_FILENAME,
        sha256: str = SILERO_VAD_SHA256,
    ) -> None:
        self.model_path = model_path
        self.repo_id = repo_id
        self.revision = revision
        self.filename = filename
        self.sha256 = sha256
        self._backend: SileroVADBackend | None = None
        self._lock = threading.Lock()

    def get(self) -> SileroVADBackend:
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is None:
                path = self._resolve_local_artifact()
                self._verify_checksum(path)
                self._backend = SileroVADBackend(path)
        return self._backend

    def _resolve_local_artifact(self) -> Path:
        if self.model_path:
            path = Path(self.model_path).expanduser()
            if path.is_file():
                return path
            raise RuntimeError(f"Configured Silero VAD model does not exist: {path}")
        cached = try_get_local_file(
            model=self.repo_id,
            file_name=self.filename,
            revision=self.revision,
        )
        if isinstance(cached, Path) and cached.is_file():
            return cached
        raise RuntimeError(
            "Silero VAD artifact is not available locally. Pre-download the pinned artifact "
            "or configure duplex_session.server_vad_model_path."
        )

    def _verify_checksum(self, path: Path) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.sha256:
            raise RuntimeError(f"Silero VAD checksum mismatch for {path}: expected {self.sha256}, got {digest}")
