# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import io
import sys
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import resample_poly

from vllm_omni.entrypoints.openai.audio_utils_mixin import StreamingAudioResampler
from vllm_omni.experimental.fullduplex.openai.audio import resample_pcm16_mono
from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexSession,
    DuplexSessionConfig,
)
from vllm_omni.experimental.fullduplex.openai.realtime_session import (
    NativeRealtimeSessionProtocol,
)
from vllm_omni.experimental.fullduplex.openai.server_vad import (
    ServerVADConfig,
    ServerVADFrame,
    ServerVADPipeline,
    SileroVADBackend,
    SileroVADBackendProvider,
    SpeechEndpointDecision,
    ThresholdEndpointPolicy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class SequenceDetector:
    frame_samples = 160

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities

    def new_state(self) -> int:
        return 0

    def infer(self, frame: np.ndarray, state: object) -> tuple[float, object]:
        index = int(state)
        probability = self.probabilities[index] if index < len(self.probabilities) else 0.0
        return probability, index + 1


def test_server_vad_config_defaults_and_validation():
    config = ServerVADConfig.from_value({"type": "server_vad"})

    assert config.type == "server_vad"
    assert config.threshold == 0.5
    assert config.prefix_padding_ms == 300
    assert config.silence_duration_ms == 500
    assert config.create_response is True
    assert config.interrupt_response is True
    assert config.min_speech_duration_ms is None
    assert config.as_dict()["type"] == config.type

    interrupting = ServerVADConfig.from_value(
        {
            "type": "server_vad",
            "interrupt_response": True,
            "min_speech_duration_ms": 96,
        }
    )
    assert interrupting.interrupt_response is True
    assert interrupting.min_speech_duration_ms == 96
    with pytest.raises(ValueError, match="Unknown"):
        ServerVADConfig.from_value({"type": "server_vad", "semantic_eagerness": "high"})


def _resample_reference(
    samples: np.ndarray,
    *,
    source_rate_hz: int = 24_000,
    target_rate_hz: int = 16_000,
) -> np.ndarray:
    normalized = samples.astype(np.float32) * np.float32(1.0 / 32768.0)
    rate_gcd = gcd(source_rate_hz, target_rate_hz)
    return np.ascontiguousarray(
        resample_poly(
            normalized,
            target_rate_hz // rate_gcd,
            source_rate_hz // rate_gcd,
        ),
        dtype=np.float32,
    )


def _process_pcm16(
    resampler: StreamingAudioResampler,
    samples: np.ndarray,
    *,
    final: bool = False,
) -> np.ndarray:
    normalized = samples.astype(np.float32) * np.float32(1.0 / 32768.0)
    return resampler.process(normalized, final=final)


def test_streaming_resampler_binds_source_and_target_rates():
    resampler = StreamingAudioResampler(24_000, 16_000)

    assert resampler.source_rate == 24_000
    assert resampler.target_rate == 16_000

    for invalid_rate in (True, 0, -1):
        with pytest.raises(ValueError, match="positive integer"):
            StreamingAudioResampler(invalid_rate, 16_000)
    with pytest.raises(ValueError, match="unsupported resampling ratio"):
        StreamingAudioResampler(191_999, 16_000)


@pytest.mark.parametrize("source_rate_hz", [8_000, 16_000, 24_000, 44_100, 48_000])
@pytest.mark.parametrize("sample_count", [1, 2, 3, 4, 5, 17, 2_399, 2_400, 2_401])
def test_streaming_resampler_matches_resample_poly(source_rate_hz: int, sample_count: int):
    rng = np.random.default_rng(sample_count)
    source = rng.integers(-32_768, 32_768, size=sample_count, dtype=np.int16)
    resampler = StreamingAudioResampler(source_rate_hz, 16_000)

    actual = _process_pcm16(resampler, source, final=True)
    expected = _resample_reference(source, source_rate_hz=source_rate_hz)

    assert actual.size == (sample_count * 16_000 + source_rate_hz - 1) // source_rate_hz
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_stateless_pcm16_resampler_matches_polyphase_reference():
    rng = np.random.default_rng(24_000)
    source = rng.integers(-32_768, 32_768, size=2_400, dtype=np.int16)

    actual = np.frombuffer(
        resample_pcm16_mono(
            source.tobytes(),
            source_rate_hz=24_000,
            target_rate_hz=16_000,
        ),
        dtype="<i2",
    )
    expected = np.clip(
        np.rint(_resample_reference(source) * 32_768.0),
        -32_768,
        32_767,
    ).astype("<i2")

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1)


def test_stateless_pcm16_resampler_bounds_unusual_rate_ratios():
    source = np.arange(2_400, dtype="<i2")

    actual = np.frombuffer(
        resample_pcm16_mono(
            source.tobytes(),
            source_rate_hz=191_999,
            target_rate_hz=16_000,
        ),
        dtype="<i2",
    )

    assert actual.size == round(source.size * 16_000 / 191_999)


@pytest.mark.parametrize(
    "chunk_sizes",
    [
        [2_401],
        [1] * 2_401,
        [2, 1] * 800 + [1],
        [7, 13, 511, 512, 513, 845],
    ],
)
def test_24khz_resampler_is_invariant_to_chunk_boundaries(chunk_sizes: list[int]):
    sample_index = np.arange(2_401, dtype=np.int32)
    source = ((sample_index * 7919) % 65_536 - 32_768).astype("<i2")
    resampler = StreamingAudioResampler(24_000, 16_000)

    chunks: list[np.ndarray] = []
    offset = 0
    for chunk_size in chunk_sizes:
        chunks.append(_process_pcm16(resampler, source[offset : offset + chunk_size]))
        chunks.append(_process_pcm16(resampler, source[:0]))
        offset += chunk_size

    assert offset == source.size
    chunks.append(_process_pcm16(resampler, source[:0], final=True))
    actual = np.concatenate(chunks)
    whole_resampler = StreamingAudioResampler(24_000, 16_000)
    whole = _process_pcm16(whole_resampler, source, final=True)
    expected = _resample_reference(source)
    np.testing.assert_array_equal(actual, whole)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert actual.size == 1_601


def test_24khz_resampler_retains_right_lookahead_until_flush():
    source = np.arange(15, dtype="<i2")
    resampler = StreamingAudioResampler(24_000, 16_000)

    assert _process_pcm16(resampler, source).size == 0

    actual = _process_pcm16(resampler, source[:0], final=True)

    np.testing.assert_allclose(actual, _resample_reference(source), rtol=2e-5, atol=2e-5)

    assert _process_pcm16(resampler, source[:0], final=True).size == 0
    with pytest.raises(RuntimeError, match="has been finalized"):
        _process_pcm16(resampler, source)


def test_24khz_resampler_reset_discards_residual_samples():
    resampler = StreamingAudioResampler(24_000, 16_000)
    discarded = np.arange(23, dtype="<i2")
    source = np.arange(64, dtype="<i2")

    _process_pcm16(resampler, discarded)
    resampler.reset()

    actual = _process_pcm16(resampler, source, final=True)

    np.testing.assert_allclose(actual, _resample_reference(source), rtol=2e-5, atol=2e-5)


def test_24khz_resampler_suppresses_content_above_16khz_nyquist():
    sample_rate_hz = 24_000
    sample_index = np.arange(sample_rate_hz, dtype=np.float64)
    source = np.rint(0.8 * 32767 * np.sin(2 * np.pi * 10_000 * sample_index / sample_rate_hz)).astype("<i2")
    resampler = StreamingAudioResampler(24_000, 16_000)

    output = _process_pcm16(resampler, source, final=True)

    # Ignore the short finite-stream edge transient. A non-filtered 3:2
    # converter aliases this tone into the output at roughly 0.4 RMS.
    steady_state_rms = float(np.sqrt(np.mean(np.square(output[1_000:]))))
    assert steady_state_rms < 0.005


def test_24khz_resampler_does_not_create_a_passband_image():
    source_rate_hz = 24_000
    target_rate_hz = 16_000
    frequency_hz = 6_000
    sample_index = np.arange(source_rate_hz * 2, dtype=np.float64)
    source = np.rint(0.8 * 32767 * np.sin(2 * np.pi * frequency_hz * sample_index / source_rate_hz)).astype("<i2")
    resampler = StreamingAudioResampler(source_rate_hz, target_rate_hz)

    output = _process_pcm16(resampler, source, final=True)[8_000:24_000]

    def amplitude_at(frequency: int) -> float:
        basis = np.exp(-2j * np.pi * frequency * np.arange(output.size) / target_rate_hz)
        return float(2 * np.abs(np.dot(output, basis)) / output.size)

    desired_amplitude = amplitude_at(frequency_hz)
    image_amplitude = amplitude_at(target_rate_hz // 2 - frequency_hz)
    desired_gain_db = 20 * np.log10(desired_amplitude / 0.8)
    image_db = 20 * np.log10(image_amplitude / desired_amplitude)

    assert abs(desired_gain_db) < 0.1
    assert image_db < -60


@pytest.mark.asyncio
async def test_server_vad_pipeline_handles_arbitrary_chunk_boundaries():
    detector = SequenceDetector([0.0, 0.9, 0.8, 0.0, 0.0])
    pipeline = ServerVADPipeline(
        detector,
        ServerVADConfig(prefix_padding_ms=10, silence_duration_ms=20),
    )
    samples = np.zeros(detector.frame_samples * 5, dtype=np.float32)

    batches = [
        await pipeline.push(samples[:73]),
        await pipeline.push(samples[73:431]),
        await pipeline.push(samples[431:]),
    ]
    frames = [frame for batch in batches for frame in batch.frames]

    assert len(frames) == 5
    assert [index for index, frame in enumerate(frames) if frame.decision.speech_started] == [1]
    assert [index for index, frame in enumerate(frames) if frame.decision.speech_stopped] == [4]

    pipeline.reset()
    reset_batch = await pipeline.push(samples)
    assert [index for index, frame in enumerate(reset_batch.frames) if frame.decision.speech_started] == [1]
    assert [index for index, frame in enumerate(reset_batch.frames) if frame.decision.speech_stopped] == [4]


@pytest.mark.asyncio
async def test_server_vad_pipeline_pcm16_resampling_is_split_invariant():
    sample_index = np.arange(2_400, dtype=np.int32)
    source = ((sample_index * 7919) % 65_536 - 32_768).astype("<i2")
    # Supply ample silence after the signal so the streaming resampler can
    # emit a complete set of VAD frames without depending on its FIR width.
    right_context = np.zeros(240, dtype="<i2")
    continuous_source = np.concatenate((source, right_context))

    async def process(chunk_sizes: list[int], *, use_bytes: bool) -> tuple[np.ndarray, list[SpeechEndpointDecision]]:
        pipeline = ServerVADPipeline(
            SequenceDetector([0.0] * 10),
            ServerVADConfig(),
        )
        frames: list[ServerVADFrame] = []
        offset = 0
        for chunk_size in chunk_sizes:
            chunk = continuous_source[offset : offset + chunk_size]
            payload = chunk.tobytes() if use_bytes else chunk
            batch = await pipeline.push_pcm16(payload, source_sample_rate_hz=24_000)
            frames.extend(batch.frames)
            offset += chunk_size
        assert offset == continuous_source.size
        return np.concatenate([frame.samples for frame in frames]), [frame.decision for frame in frames]

    whole_samples, whole_decisions = await process([continuous_source.size], use_bytes=False)
    split_samples, split_decisions = await process(
        [7, 13, 511, 512, 513, continuous_source.size - 1_556],
        use_bytes=True,
    )

    np.testing.assert_array_equal(split_samples, whole_samples)
    np.testing.assert_allclose(whole_samples, _resample_reference(source), rtol=2e-5, atol=2e-5)
    assert split_decisions == whole_decisions


@pytest.mark.asyncio
async def test_server_vad_pipeline_empty_append_does_not_bind_source_rate():
    pipeline = ServerVADPipeline(
        SequenceDetector([0.0]),
        ServerVADConfig(),
    )

    empty_batch = await pipeline.push_pcm16(b"", source_sample_rate_hz=24_000)
    assert not empty_batch.frames

    batch = await pipeline.push_pcm16(
        np.zeros(160, dtype="<i2"),
        source_sample_rate_hz=16_000,
    )
    assert len(batch.frames) == 1


@pytest.mark.asyncio
async def test_server_vad_pipeline_reset_clears_resampler_and_source_rate_lock():
    pipeline = ServerVADPipeline(
        SequenceDetector([0.0]),
        ServerVADConfig(),
    )

    batch = await pipeline.push_pcm16(
        np.asarray([30_000, -30_000], dtype="<i2"),
        source_sample_rate_hz=24_000,
    )
    assert not batch.frames

    pipeline.reset()

    source = np.zeros(160, dtype="<i2")
    source[:3] = [-32768, 0, 32767]
    batch = await pipeline.push_pcm16(source.tobytes(), source_sample_rate_hz=16_000)

    assert len(batch.frames) == 1
    np.testing.assert_array_equal(
        batch.frames[0].samples[:3],
        np.asarray([-1.0, 0.0, 32767 / 32768], dtype=np.float32),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("first_rate,second_rate", [(16_000, 24_000), (24_000, 16_000)])
async def test_server_vad_pipeline_rejects_source_rate_switch(first_rate: int, second_rate: int):
    pipeline = ServerVADPipeline(
        SequenceDetector([0.0]),
        ServerVADConfig(),
    )

    await pipeline.push_pcm16(np.asarray([1], dtype="<i2"), source_sample_rate_hz=first_rate)

    with pytest.raises(ValueError, match="cannot change"):
        await pipeline.push_pcm16(np.asarray([2], dtype="<i2"), source_sample_rate_hz=second_rate)


def test_threshold_endpoint_policy_uses_silero_v62_exit_hysteresis():
    policy = ThresholdEndpointPolicy(
        ServerVADConfig(threshold=0.5, silence_duration_ms=20),
        sample_rate_hz=16_000,
    )
    frame_samples = 160

    started = policy.update(0.8, frame_start_sample=0, frame_samples=frame_samples)
    assert started.speech_started is True

    # Values below the activation threshold but above threshold - 0.15 must
    # not begin a silence candidate or close an active speech segment.
    for frame_index in range(1, 5):
        decision = policy.update(
            0.4,
            frame_start_sample=frame_index * frame_samples,
            frame_samples=frame_samples,
        )
        assert decision.speech_stopped is False

    first_silence = policy.update(
        0.2,
        frame_start_sample=5 * frame_samples,
        frame_samples=frame_samples,
    )
    stopped = policy.update(
        0.2,
        frame_start_sample=6 * frame_samples,
        frame_samples=frame_samples,
    )

    assert first_silence.speech_stopped is False
    assert stopped.speech_stopped is True


def test_threshold_endpoint_policy_honors_minimum_speech_duration():
    policy = ThresholdEndpointPolicy(
        ServerVADConfig(prefix_padding_ms=0, min_speech_duration_ms=25),
        sample_rate_hz=16_000,
    )
    frame_samples = 160

    assert not policy.update(0.9, frame_start_sample=0, frame_samples=frame_samples).speech_started
    assert not policy.update(0.9, frame_start_sample=160, frame_samples=frame_samples).speech_started
    started = policy.update(0.9, frame_start_sample=320, frame_samples=frame_samples)

    assert started.speech_started is True
    assert started.audio_start_ms == 0


def test_threshold_endpoint_policy_clamps_silero_exit_threshold():
    policy = ThresholdEndpointPolicy(
        ServerVADConfig(threshold=0.1, silence_duration_ms=20),
        sample_rate_hz=16_000,
    )
    frame_samples = 160

    started = policy.update(0.9, frame_start_sample=0, frame_samples=frame_samples)
    assert started.speech_started is True

    first_silence = policy.update(
        0.0,
        frame_start_sample=frame_samples,
        frame_samples=frame_samples,
    )
    stopped = policy.update(
        0.0,
        frame_start_sample=2 * frame_samples,
        frame_samples=frame_samples,
    )

    assert first_silence.speech_stopped is False
    assert stopped.speech_stopped is True
    # The committed audio ends at 30 ms: 10 ms of speech followed by the
    # 20 ms of silence required to detect the endpoint.
    assert stopped.audio_end_ms == 30
    assert stopped.endpoint_delay_ms == 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("audio", "part_overrides"),
    [
        pytest.param("", {}, id="empty"),
        pytest.param("AAAAAA==!!!!", {}, id="invalid-base64"),
        pytest.param(
            base64.b64encode(np.zeros(480, dtype="<i2").tobytes()).decode(),
            {"sample_rate_hz": 191_999},
            id="sample-rate-override",
        ),
        pytest.param(
            base64.b64encode(np.zeros(480, dtype="<i2").tobytes()).decode(),
            {"format": "pcm_f32le"},
            id="format-override",
        ),
    ],
)
async def test_invalid_complete_audio_item_returns_correlated_bad_audio(
    audio: str,
    part_overrides: dict[str, object],
):
    protocol = NativeRealtimeSessionProtocol({})
    protocol._turn_detection_configured = True
    protocol._turn_detection = ServerVADConfig().as_dict()
    protocol._hold_realtime_output_until_session_created = False
    sent: list[dict[str, object]] = []

    async def send(payload: dict[str, object]) -> None:
        sent.append(payload)

    protocol.bind_sender(send)

    translated = await protocol._to_duplex_event(
        {
            "type": "conversation.item.create",
            "event_id": "event-invalid-audio",
            "item": {
                "id": "item-invalid-audio",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_audio", "audio": audio, **part_overrides}],
            },
        }
    )

    assert translated is None
    assert sent[-1]["type"] == "error"
    assert sent[-1]["error"]["code"] == "bad_audio"
    assert sent[-1]["error"]["event_id"] == "event-invalid-audio"
    assert not {"conversation.item.added", "conversation.item.done"}.intersection(
        payload.get("type") for payload in sent
    )


def test_silero_backend_matches_upstream_v62_streaming_contract(monkeypatch, tmp_path):
    class FakeSessionOptions:
        inter_op_num_threads = 0
        intra_op_num_threads = 0

    class FakeInput:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeInferenceSession:
        instances: list[FakeInferenceSession] = []

        def __init__(self, path: str, *, providers: list[str], sess_options: object) -> None:
            self.path = path
            self.providers = providers
            self.sess_options = sess_options
            self.calls: list[dict[str, np.ndarray]] = []
            self.run_barrier: threading.Barrier | None = None
            self.instances.append(self)

        def get_inputs(self) -> list[FakeInput]:
            return [FakeInput("input"), FakeInput("state"), FakeInput("sr")]

        def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            self.calls.append({name: np.array(value, copy=True) for name, value in inputs.items()})
            if self.run_barrier is not None:
                self.run_barrier.wait(timeout=5)
            return [
                np.asarray([[0.75]], dtype=np.float32),
                np.full((2, 1, 128), len(self.calls), dtype=np.float32),
            ]

    fake_ort = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        InferenceSession=FakeInferenceSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    model_path = tmp_path / "silero_vad.onnx"
    model_path.write_bytes(b"fake-model")
    backend = SileroVADBackend(model_path)
    session = FakeInferenceSession.instances[0]

    assert session.providers == ["CPUExecutionProvider"]
    assert session.sess_options.inter_op_num_threads == 1
    assert session.sess_options.intra_op_num_threads == 1
    assert session.calls[0]["input"].shape == (1, 576)
    assert session.calls[0]["state"].shape == (2, 1, 128)
    np.testing.assert_array_equal(session.calls[0]["state"], np.zeros((2, 1, 128), dtype=np.float32))
    assert session.calls[0]["sr"].shape == ()
    assert session.calls[0]["sr"].item() == 16_000

    state = backend.new_state()
    first_frame = np.arange(backend.frame_samples, dtype=np.float32)
    probability, state = backend.infer(first_frame, state)
    first_call = session.calls[1]

    assert probability == pytest.approx(0.75)
    np.testing.assert_array_equal(
        first_call["input"][:, : backend.context_samples],
        np.zeros((1, backend.context_samples), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        first_call["input"][:, backend.context_samples :],
        first_frame[None, :],
    )
    np.testing.assert_array_equal(first_call["state"], np.zeros((2, 1, 128), dtype=np.float32))

    second_frame = -first_frame
    backend.infer(second_frame, state)
    second_call = session.calls[2]
    np.testing.assert_array_equal(
        second_call["input"][:, : backend.context_samples],
        first_frame[None, -backend.context_samples :],
    )
    np.testing.assert_array_equal(
        second_call["input"][:, backend.context_samples :],
        second_frame[None, :],
    )
    np.testing.assert_array_equal(
        second_call["state"],
        np.full((2, 1, 128), 2, dtype=np.float32),
    )

    # Independent pipeline states may enter the shared ORT session concurrently.
    session.run_barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(backend.infer, first_frame, backend.new_state()),
            executor.submit(backend.infer, second_frame, backend.new_state()),
        ]
        assert [future.result()[0] for future in futures] == pytest.approx([0.75, 0.75])


def test_duplex_session_owns_server_vad_prefix_and_utterance_audio():
    session = DuplexSession("server-vad-session", DuplexSessionConfig())
    frame = np.zeros(160, dtype=np.float32)

    session.reserve_input_bytes(frame.nbytes * 3, limit=frame.nbytes * 4)
    released = session.append_server_vad_frame(
        frame,
        speech_started=False,
        speech_stopped=False,
        prefix_samples=160,
    )
    assert released == 0
    released = session.append_server_vad_frame(
        frame,
        speech_started=False,
        speech_stopped=False,
        prefix_samples=160,
    )
    assert released == frame.nbytes
    session.release_input_bytes(released)
    session.append_server_vad_frame(
        frame,
        speech_started=True,
        speech_stopped=False,
        prefix_samples=160,
    )

    assert session.server_vad_utterance_bytes == frame.nbytes * 2
    assert session.stage_server_vad_audio_for_commit() is True
    committed = session.commit_user_input()
    assert committed is not None
    assert committed.message["role"] == "user"
    audio_url = committed.message["content"][0]["audio_url"]["url"]
    assert audio_url.startswith("data:audio/wav;base64,")
    wav_bytes = base64.b64decode(audio_url.partition(",")[2])
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnframes() == frame.size * 2


def test_silero_provider_rejects_missing_or_invalid_local_artifact(tmp_path):
    missing = SileroVADBackendProvider(model_path=str(tmp_path / "missing.onnx"))
    with pytest.raises(RuntimeError, match="does not exist"):
        missing.get()

    invalid_path = tmp_path / "silero_vad.onnx"
    invalid_path.write_bytes(b"not-the-pinned-model")
    invalid = SileroVADBackendProvider(
        model_path=str(invalid_path),
        sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        invalid.get()
