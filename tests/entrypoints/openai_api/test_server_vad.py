# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import io
import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest

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
    _PCM16Mono24kTo16kResampler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class SequenceDetector:
    frame_samples = 160

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.new_state_calls = 0

    def new_state(self) -> int:
        self.new_state_calls += 1
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
    assert config.interrupt_response is False
    assert config.as_dict()["type"] == config.type

    with pytest.raises(ValueError, match="interrupt_response"):
        ServerVADConfig.from_value({"type": "server_vad", "interrupt_response": True})
    with pytest.raises(ValueError, match="Unknown"):
        ServerVADConfig.from_value({"type": "server_vad", "semantic_eagerness": "high"})


def _resample_24khz_reference(samples: np.ndarray) -> np.ndarray:
    normalized = samples.astype(np.float32) * np.float32(1.0 / 32768.0)
    taps = _PCM16Mono24kTo16kResampler._design_lowpass()
    history = np.zeros(taps.size - 1, dtype=np.float32)
    filtered = np.convolve(np.concatenate((history, normalized)), taps, mode="valid").astype(
        np.float32,
        copy=False,
    )
    complete_samples = filtered.size - filtered.size % 3
    triplets = filtered[:complete_samples].reshape(-1, 3)
    expected = np.empty(triplets.shape[0] * 2, dtype=np.float32)
    expected[0::2] = triplets[:, 0]
    expected[1::2] = (triplets[:, 1] + triplets[:, 2]) * np.float32(0.5)
    return expected


@pytest.mark.parametrize(
    "chunk_sizes",
    [
        [2_400],
        [1] * 2_400,
        [2, 1] * 800,
        [7, 13, 511, 512, 513, 844],
    ],
)
def test_24khz_resampler_is_invariant_to_chunk_boundaries(chunk_sizes: list[int]):
    sample_index = np.arange(2_400, dtype=np.int32)
    source = ((sample_index * 7919) % 65_536 - 32_768).astype("<i2")
    resampler = _PCM16Mono24kTo16kResampler()

    chunks: list[np.ndarray] = []
    offset = 0
    for chunk_size in chunk_sizes:
        chunks.append(resampler.push(source[offset : offset + chunk_size]))
        offset += chunk_size

    assert offset == source.size
    actual = np.concatenate(chunks)
    expected = _resample_24khz_reference(source)
    np.testing.assert_array_equal(actual, expected)
    assert actual.size == 1_600
    assert resampler.pending_samples == 0


def test_24khz_resampler_retains_incomplete_triplet():
    source = np.asarray([-32768, 12_000, 20_000], dtype="<i2")
    resampler = _PCM16Mono24kTo16kResampler()

    assert resampler.push(source[:1]).size == 0
    assert resampler.pending_samples == 1
    assert resampler.scratch_bytes == 4
    assert resampler.push(source[1:2]).size == 0
    assert resampler.pending_samples == 2
    assert resampler.scratch_bytes == 8

    actual = resampler.push(source[2:])

    np.testing.assert_array_equal(actual, _resample_24khz_reference(source))
    assert resampler.pending_samples == 0
    assert resampler.scratch_bytes == 0


def test_24khz_resampler_reset_discards_residual_samples():
    resampler = _PCM16Mono24kTo16kResampler()
    discarded = np.asarray([30_000, -30_000], dtype="<i2")
    source = np.asarray([-32768, 0, 32767], dtype="<i2")

    assert resampler.push(discarded).size == 0
    assert resampler.pending_samples == 2
    resampler.reset()

    actual = resampler.push(source)

    np.testing.assert_array_equal(actual, _resample_24khz_reference(source))
    assert resampler.pending_samples == 0


def test_24khz_resampler_suppresses_content_above_16khz_nyquist():
    sample_rate_hz = 24_000
    sample_index = np.arange(sample_rate_hz, dtype=np.float64)
    source = np.rint(0.8 * 32767 * np.sin(2 * np.pi * 10_000 * sample_index / sample_rate_hz)).astype("<i2")
    resampler = _PCM16Mono24kTo16kResampler()

    output = resampler.push(source)

    # Ignore the short causal FIR startup transient. A non-filtered 3:2
    # converter aliases this tone into the output at roughly 0.4 RMS.
    steady_state_rms = float(np.sqrt(np.mean(np.square(output[1_000:]))))
    assert steady_state_rms < 0.005


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
    assert detector.new_state_calls == 1
    assert pipeline.scratch_bytes == 0

    pipeline.reset()
    assert detector.new_state_calls == 2


@pytest.mark.asyncio
async def test_server_vad_pipeline_pcm16_resampling_is_split_invariant():
    sample_index = np.arange(2_400, dtype=np.int32)
    source = ((sample_index * 7919) % 65_536 - 32_768).astype("<i2")

    async def process(chunk_sizes: list[int], *, use_bytes: bool) -> tuple[np.ndarray, list[SpeechEndpointDecision]]:
        pipeline = ServerVADPipeline(
            SequenceDetector([0.0] * 10),
            ServerVADConfig(),
        )
        frames: list[ServerVADFrame] = []
        offset = 0
        for chunk_size in chunk_sizes:
            chunk = source[offset : offset + chunk_size]
            payload = chunk.tobytes() if use_bytes else chunk
            batch = await pipeline.push_pcm16(payload, source_sample_rate_hz=24_000)
            frames.extend(batch.frames)
            offset += chunk_size
        assert offset == source.size
        assert pipeline.scratch_bytes == 0
        return np.concatenate([frame.samples for frame in frames]), [frame.decision for frame in frames]

    whole_samples, whole_decisions = await process([2_400], use_bytes=False)
    split_samples, split_decisions = await process(
        [7, 13, 511, 512, 513, 844],
        use_bytes=True,
    )

    np.testing.assert_array_equal(split_samples, whole_samples)
    np.testing.assert_array_equal(whole_samples, _resample_24khz_reference(source))
    assert split_decisions == whole_decisions


@pytest.mark.asyncio
async def test_server_vad_pipeline_reset_clears_resampler_and_source_rate_lock():
    pipeline = ServerVADPipeline(
        SequenceDetector([0.0]),
        ServerVADConfig(),
    )

    batch = await pipeline.push_pcm16(np.asarray([30_000, -30_000], dtype="<i2"), source_sample_rate_hz=24_000)
    assert not batch.frames
    assert pipeline.scratch_bytes == 8

    pipeline.reset()
    assert pipeline.scratch_bytes == 0

    source = np.zeros(160, dtype="<i2")
    source[:3] = [-32768, 0, 32767]
    batch = await pipeline.push_pcm16(source.tobytes(), source_sample_rate_hz=16_000)

    assert len(batch.frames) == 1
    np.testing.assert_array_equal(
        batch.frames[0].samples[:3],
        np.asarray([-1.0, 0.0, 32767 / 32768], dtype=np.float32),
    )
    assert pipeline.scratch_bytes == 0


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
        assert policy.speech_active is True

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
    assert policy.speech_active is False


@pytest.mark.asyncio
async def test_server_vad_complete_audio_item_bypasses_input_buffer_and_normalizes_to_wav():
    protocol = NativeRealtimeSessionProtocol({})
    protocol._turn_detection_configured = True
    protocol._turn_detection = ServerVADConfig().as_dict()
    protocol._input_audio_format = "pcm16"
    protocol._input_sample_rate_hz = 24_000
    pcm16 = np.arange(2_400, dtype="<i2")

    public_audio = base64.b64encode(pcm16.tobytes()).decode()
    event = {
        "type": "conversation.item.create",
        "event_id": "event-full-audio",
        "item": {
            "id": "item-full-audio",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "audio": public_audio}],
        },
    }

    translated = await protocol._to_duplex_event(event)

    assert translated is not None
    assert translated["type"] == "turn.signal"
    assert translated["event"] == "conversation.item.create"
    public_item = translated["payload"]["item"]
    assert public_item["content"] == [{"type": "input_audio", "audio": public_audio}]
    history_item = translated["payload"]["history_item"]
    part = history_item["content"][0]
    assert part["type"] == "input_audio"
    assert part["format"] == "wav"
    assert part["sample_rate_hz"] == 16_000
    with wave.open(io.BytesIO(base64.b64decode(part["audio"])), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnframes() == 1_600

    assert protocol._pending_outbound.empty()
    assert protocol._input_speech_started is False
    assert protocol._input_audio_buffer_has_audio is False
    assert protocol._turn_detection_config_locked is False


@pytest.mark.asyncio
async def test_server_vad_complete_audio_item_keeps_wire_item_unchanged():
    protocol = NativeRealtimeSessionProtocol({})
    protocol._turn_detection_configured = True
    protocol._turn_detection = ServerVADConfig().as_dict()
    protocol._input_audio_format = "pcm16"
    protocol._input_sample_rate_hz = 16_000
    protocol._hold_realtime_output_until_session_created = False
    sent: list[dict[str, object]] = []

    async def send(payload: dict[str, object]) -> None:
        sent.append(payload)

    protocol.bind_sender(send)
    public_audio = base64.b64encode(np.arange(320, dtype="<i2").tobytes()).decode()
    event = {
        "type": "conversation.item.create",
        "event_id": "event-public-audio",
        "item": {
            "id": "item-public-audio",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "audio": public_audio}],
        },
    }

    translated = await protocol._to_duplex_event(event)
    assert translated is not None
    await protocol._send_realtime_input_ack(event)
    projected = protocol.encode_outbound_event(
        {
            "type": "conversation.item.created",
            "item": translated["payload"]["item"],
            "created": True,
        }
    )

    wire_items = [
        payload["item"]
        for payload in [*sent, *projected]
        if payload["type"]
        in {
            "conversation.item.added",
            "conversation.item.created",
            "conversation.item.done",
        }
    ]
    assert len(wire_items) == 3
    assert all(item == wire_items[0] for item in wire_items)
    assert wire_items[0]["content"] == [{"type": "input_audio", "audio": public_audio}]


@pytest.mark.asyncio
async def test_empty_complete_audio_item_returns_bad_audio_without_registration():
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
            "event_id": "event-empty-audio",
            "item": {
                "id": "item-empty-audio",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_audio", "audio": ""}],
            },
        }
    )

    assert translated is None
    assert sent[-1]["type"] == "error"
    assert sent[-1]["error"]["code"] == "bad_audio"
    assert "item-empty-audio" not in protocol._conversation_items


@pytest.mark.asyncio
async def test_complete_audio_item_preserves_manual_commit_translation_without_server_vad():
    protocol = NativeRealtimeSessionProtocol({})
    pcm16 = np.full(320, 8_192, dtype="<i2")

    translated = await protocol._conversation_item_to_duplex(
        {
            "type": "conversation.item.create",
            "item": {
                "id": "item-manual-audio",
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "audio": base64.b64encode(pcm16.tobytes()).decode(),
                    }
                ],
            },
        }
    )

    assert translated is not None
    assert translated["type"] == "input_audio_buffer.append"
    queued = protocol._pending_outbound.get_nowait()
    assert queued["type"] == "input_audio_buffer.commit"
    assert queued["item_id"] == "item-manual-audio"


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
            self.instances.append(self)

        def get_inputs(self) -> list[FakeInput]:
            return [FakeInput("input"), FakeInput("state"), FakeInput("sr")]

        def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            self.calls.append({name: np.array(value, copy=True) for name, value in inputs.items()})
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
