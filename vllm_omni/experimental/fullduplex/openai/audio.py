# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import binascii
import io
import math
import wave

import numpy as np

try:
    from audioop import alaw2lin, lin2alaw, lin2ulaw, ulaw2lin
except ImportError:  # pragma: no cover - audioop is removed in newer Python.
    alaw2lin = lin2alaw = lin2ulaw = ulaw2lin = None


MIN_INPUT_SAMPLE_RATE_HZ = 8_000
MAX_INPUT_SAMPLE_RATE_HZ = 192_000


def validate_input_sample_rate_hz(sample_rate_hz: int | float) -> int:
    if isinstance(sample_rate_hz, bool) or (isinstance(sample_rate_hz, float) and not math.isfinite(sample_rate_hz)):
        raise ValueError("sample_rate_hz must be a finite integer")
    rate = int(sample_rate_hz)
    if rate != sample_rate_hz:
        raise ValueError("sample_rate_hz must be an integer")
    if not MIN_INPUT_SAMPLE_RATE_HZ <= rate <= MAX_INPUT_SAMPLE_RATE_HZ:
        raise ValueError(f"sample_rate_hz must be between {MIN_INPUT_SAMPLE_RATE_HZ} and {MAX_INPUT_SAMPLE_RATE_HZ}")
    return rate


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
    _max_polyphase_factor = 2_048

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
        if max(self._up, self._down) > self._max_polyphase_factor:
            raise ValueError("source and target sample rates have an unsupported resampling ratio")
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


def resample_pcm16_mono(raw: bytes, *, source_rate_hz: int, target_rate_hz: int) -> bytes:
    if source_rate_hz <= 0 or target_rate_hz <= 0 or source_rate_hz == target_rate_hz:
        return raw
    samples = np.frombuffer(raw, dtype="<i2")
    rate_gcd = math.gcd(source_rate_hz, target_rate_hz)
    reduced_factor = max(source_rate_hz // rate_gcd, target_rate_hz // rate_gcd)
    if reduced_factor > _StreamingPCM16MonoResampler._max_polyphase_factor:
        # The FIR size scales with the reduced ratio. Keep unusual client
        # rates bounded while common audio rates use the high-quality path.
        if samples.size <= 1:
            return raw
        target_size = max(1, int(round(samples.size * target_rate_hz / source_rate_hz)))
        source_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=True)
        target_x = np.linspace(0.0, 1.0, num=target_size, endpoint=True)
        resampled = np.interp(target_x, source_x, samples)
        return np.clip(resampled, -32_768, 32_767).astype("<i2").tobytes()
    resampler = _StreamingPCM16MonoResampler(
        source_rate_hz=source_rate_hz,
        target_rate_hz=target_rate_hz,
    )
    resampled = np.concatenate((resampler.push(samples), resampler.flush()))
    return np.clip(np.rint(resampled * 32768.0), -32768, 32767).astype("<i2").tobytes()


def decode_g711_ulaw(raw: bytes) -> bytes:
    if ulaw2lin is not None:
        return ulaw2lin(raw, 2)
    data = np.frombuffer(raw, dtype=np.uint8)
    value = np.bitwise_not(data).astype(np.int16)
    sign = value & 0x80
    exponent = (value >> 4) & 0x07
    mantissa = value & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    return np.where(sign != 0, -(sample - 0x84), sample - 0x84).astype("<i2").tobytes()


def decode_g711_alaw(raw: bytes) -> bytes:
    if alaw2lin is not None:
        return alaw2lin(raw, 2)
    data = np.bitwise_xor(np.frombuffer(raw, dtype=np.uint8), 0x55).astype(np.int16)
    sign = data & 0x80
    exponent = (data >> 4) & 0x07
    mantissa = data & 0x0F
    sample = np.where(exponent == 0, (mantissa << 4) + 8, ((mantissa << 4) + 0x108) << (exponent - 1))
    return np.where(sign != 0, sample, -sample).astype("<i2").tobytes()


def encode_g711_ulaw(raw: bytes) -> bytes:
    if lin2ulaw is not None:
        return lin2ulaw(raw, 2)
    pcm = np.clip(np.frombuffer(raw, dtype="<i2").astype(np.int32), -32635, 32635)
    sign = np.where(pcm < 0, 0x80, 0)
    magnitude = np.abs(pcm) + 0x84
    exponent = np.zeros_like(magnitude)
    for exp in range(7):
        exponent = np.where(magnitude > (0xFF << exp), exp + 1, exponent)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (np.bitwise_not(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


def encode_g711_alaw(raw: bytes) -> bytes:
    if lin2alaw is not None:
        return lin2alaw(raw, 2)
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    sign = np.where(pcm >= 0, 0x80, 0x00)
    magnitude = np.abs(pcm)
    exponent = np.zeros_like(magnitude)
    for exp in range(1, 8):
        exponent = np.where(magnitude >= (1 << (exp + 7)), exp, exponent)
    mantissa = np.where(
        exponent == 0,
        (magnitude >> 4) & 0x0F,
        (magnitude >> (exponent + 3)) & 0x0F,
    )
    return ((sign | (exponent << 4) | mantissa) ^ 0x55).astype(np.uint8).tobytes()


def wav_payload_to_pcm16(raw: bytes) -> tuple[bytes | None, int | None]:
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate_hz = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
        if sample_width != 2:
            return None, sample_rate_hz
        if channels <= 1:
            return frames, sample_rate_hz
        pcm = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
        mono = np.mean(pcm.astype(np.float32), axis=1)
        return np.clip(mono, -32768, 32767).astype("<i2").tobytes(), sample_rate_hz
    except (EOFError, ValueError, wave.Error):
        return None, None


def encode_float32_mono_wav_base64(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
) -> str:
    """Package normalized mono float32 samples as a PCM16 WAV data payload."""
    rate = validate_input_sample_rate_hz(sample_rate_hz)
    normalized = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
    pcm16 = np.clip(np.rint(normalized * 32768.0), -32768, 32767).astype("<i2")
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(rate)
            wav_file.writeframes(pcm16.tobytes())
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def convert_input_audio_with_rate(
    audio: object,
    fmt: object,
    *,
    sample_rate_hz: int | float | None = None,
    target_sample_rate_hz: int = 16_000,
) -> tuple[object, object, int | float | None]:
    if not isinstance(audio, str) or not isinstance(fmt, str):
        return audio, fmt, sample_rate_hz
    normalized = fmt.lower()
    if normalized not in {"pcm16", "pcm_s16le", "s16le", "g711_ulaw", "g711_alaw"}:
        return audio, fmt, sample_rate_hz
    if isinstance(sample_rate_hz, int | float):
        sample_rate_hz = validate_input_sample_rate_hz(sample_rate_hz)
    try:
        raw = base64.b64decode(audio.strip(), validate=False)
    except (binascii.Error, ValueError):
        return audio, fmt, sample_rate_hz
    if normalized == "g711_ulaw":
        raw = decode_g711_ulaw(raw)
        sample_rate_hz = sample_rate_hz if isinstance(sample_rate_hz, int | float) else 8_000
    elif normalized == "g711_alaw":
        raw = decode_g711_alaw(raw)
        sample_rate_hz = sample_rate_hz if isinstance(sample_rate_hz, int | float) else 8_000
    elif len(raw) % 2:
        return audio, fmt, sample_rate_hz
    if isinstance(sample_rate_hz, int | float) and int(sample_rate_hz) != target_sample_rate_hz:
        raw = resample_pcm16_mono(
            raw,
            source_rate_hz=int(sample_rate_hz),
            target_rate_hz=target_sample_rate_hz,
        )
        sample_rate_hz = target_sample_rate_hz
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    encoded = base64.b64encode(np.ascontiguousarray(pcm, dtype="<f4").tobytes()).decode("ascii")
    return encoded, "pcm_f32le", sample_rate_hz


def convert_output_audio(
    audio: str,
    *,
    source_fmt: str,
    target_fmt: str,
    source_sample_rate_hz: int | None = None,
    target_sample_rate_hz: int | None = None,
) -> tuple[str, str, int | None]:
    target = target_fmt.lower()
    if target not in {"g711_ulaw", "g711_alaw"}:
        return audio, source_fmt, source_sample_rate_hz
    try:
        raw = base64.b64decode(audio, validate=False)
    except (binascii.Error, ValueError):
        return audio, source_fmt, source_sample_rate_hz
    source = source_fmt.lower()
    if source == "wav":
        pcm_raw, wav_rate = wav_payload_to_pcm16(raw)
        if pcm_raw is None:
            return audio, source_fmt, source_sample_rate_hz
        raw = pcm_raw
        source_sample_rate_hz = source_sample_rate_hz or wav_rate
    elif source not in {"pcm", "pcm16", "pcm_s16le", "s16le"} or len(raw) % 2:
        return audio, source_fmt, source_sample_rate_hz
    target_rate = target_sample_rate_hz or 8_000
    if source_sample_rate_hz is not None:
        raw = resample_pcm16_mono(raw, source_rate_hz=source_sample_rate_hz, target_rate_hz=target_rate)
    encoded = encode_g711_ulaw(raw) if target == "g711_ulaw" else encode_g711_alaw(raw)
    return base64.b64encode(encoded).decode("ascii"), target, target_rate
