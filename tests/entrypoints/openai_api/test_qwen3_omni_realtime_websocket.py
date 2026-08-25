# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online tests for Qwen3-Omni /v1/realtime WebSocket (streaming PCM in, audio out).

Four scenarios:
- Ready CI: async_chunk on, smoke only (no send delay, no accuracy check).
- Merge CI: async_chunk on + send delay, full accuracy check.
- Merge CI: async_chunk off, no send delay, full accuracy check.
- Merge CI: server VAD, two turns without client commits.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import wave

import pytest
import websockets

from tests.helpers.mark import hardware_test
from tests.helpers.media import (
    convert_audio_bytes_to_text,
    cosine_similarity_text,
    generate_synthetic_audio,
)
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.experimental.fullduplex.openai.server_vad import (
    SILERO_VAD_FILENAME,
    SILERO_VAD_REPO_ID,
    SILERO_VAD_REVISION,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Synthetic input for realtime E2E (``generate_synthetic_audio``); distinct cache file per phrase.
REALTIME_SYNTH_PHRASE_TEXT = (
    "Translate into Chinese: Beijing is the Capital of China. It is the center of culture and politics"
)

# Simulate realtime upload pacing (``openai_realtime_client.py --send-delay-ms``).
SEND_DELAY_MS = 200

# CI overlay bakes in async_chunk: False and covers CUDA/ROCm/XPU via ``platforms:``.
default_stage_config = get_deploy_config_path("ci/qwen3_omni_moe.yaml")

realtime_sync_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=default_stage_config,
            use_stage_cli=True,
        ),
        id="sync",
    ),
]

realtime_async_chunk_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=default_stage_config,
            use_stage_cli=True,
            server_args=["--async-chunk"],
        ),
        id="async_chunk",
    ),
]


def _pcm16_mono_16k_from_wav_bytes(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono WAV, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM, sampwidth={wf.getsampwidth()}")
        if wf.getframerate() != 16000:
            raise ValueError(f"Expected 16 kHz input for /v1/realtime, got {wf.getframerate()} Hz")
        if wf.getcomptype() != "NONE":
            raise ValueError(f"Expected uncompressed PCM, comptype={wf.getcomptype()!r}")
        return wf.readframes(wf.getnframes())


def _wav_bytes_from_pcm16(pcm: bytes, sample_rate_hz: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm)
    return buf.getvalue()


async def _run_realtime_audio_roundtrip(
    host: str,
    port: int,
    model: str,
    pcm16: bytes,
    *,
    chunk_ms: int = 100,
    send_delay_ms: int = 0,
    server_vad: bool = False,
    turns: int = 1,
) -> dict:
    uri = f"ws://{host}:{port}/v1/realtime"
    if not server_vad:
        uri += "?duplex=0"
    incremental: list[bytes] = []
    output_sr = 24000
    text_chunks: list[str] = []
    final_text = ""
    delta_events = 0
    committed_events = 0
    turn_results: list[dict[str, object]] = []

    bytes_per_ms = 16000 * 2 // 1000
    chunk_bytes = max(bytes_per_ms * chunk_ms, 2)

    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        if server_vad:
            session_update = {
                "type": "session.update",
                "session": {
                    "model": model,
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 16_000},
                            "turn_detection": {
                                "type": "server_vad",
                                "silence_duration_ms": 1_000,
                                "create_response": True,
                                "interrupt_response": False,
                            },
                        }
                    },
                },
            }
        else:
            session_update = {"type": "session.update", "model": model}
        await ws.send(json.dumps(session_update))

        for turn_index in range(1, turns + 1):
            turn_audio: list[bytes] = []
            turn_text: list[str] = []
            turn_transcript: list[str] = []
            turn_final_text = ""
            turn_final_transcript = ""
            turn_event_types: list[str] = []
            turn_delta_events = 0
            turn_committed_events = 0
            turn_output_sr = 24_000
            speech_started_item_id: str | None = None
            speech_stopped_item_id: str | None = None
            input_item_id: str | None = None
            response_id: str | None = None
            response_status: str | None = None

            if not server_vad:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))

            for i in range(0, len(pcm16), chunk_bytes):
                chunk = pcm16[i : i + chunk_bytes]
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("utf-8"),
                        }
                    )
                )
                if send_delay_ms > 0:
                    await asyncio.sleep(send_delay_ms / 1000.0)

            if server_vad:
                silence = bytes(16_000 * 2 * 3 // 2)
                for i in range(0, len(silence), chunk_bytes):
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(silence[i : i + chunk_bytes]).decode("utf-8"),
                            }
                        )
                    )
            else:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=600)
                if isinstance(message, bytes):
                    continue

                event = json.loads(message)
                event_type = event.get("type")

                if server_vad:
                    event_type = str(event_type or "")
                    turn_event_types.append(event_type)

                    if event_type == "error":
                        raise AssertionError(f"WebSocket error: {event}")

                    if event_type == "input_audio_buffer.committed":
                        committed_events += 1
                        turn_committed_events += 1
                        item_id = event.get("item_id")
                        if isinstance(item_id, str):
                            input_item_id = item_id
                        continue

                    if event_type == "input_audio_buffer.speech_started":
                        item_id = event.get("item_id")
                        if isinstance(item_id, str):
                            speech_started_item_id = item_id
                        continue

                    if event_type == "input_audio_buffer.speech_stopped":
                        item_id = event.get("item_id")
                        if isinstance(item_id, str):
                            speech_stopped_item_id = item_id
                        continue

                    if event_type == "response.created":
                        response = event.get("response")
                        if isinstance(response, dict) and isinstance(response.get("id"), str):
                            response_id = response["id"]
                        continue

                    if event_type == "response.audio.delta":
                        assert response_id is not None
                        event_response_id = event.get("response_id")
                        assert event_response_id == response_id
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            pcm_delta = base64.b64decode(delta)
                            turn_audio.append(pcm_delta)
                            incremental.append(pcm_delta)
                            turn_delta_events += 1
                            delta_events += 1
                        sample_rate_hz = event.get("sample_rate_hz")
                        if isinstance(sample_rate_hz, int) and sample_rate_hz > 0:
                            turn_output_sr = sample_rate_hz
                            output_sr = sample_rate_hz
                        continue

                    if event_type == "response.output_text.delta":
                        assert event.get("response_id") == response_id
                        delta = event.get("delta")
                        if isinstance(delta, str):
                            turn_text.append(delta)
                            text_chunks.append(delta)
                        continue

                    if event_type == "response.output_text.done":
                        assert event.get("response_id") == response_id
                        text = event.get("text")
                        if isinstance(text, str):
                            turn_final_text = text
                            final_text = text
                        continue

                    if event_type == "response.audio_transcript.delta":
                        assert event.get("response_id") == response_id
                        delta = event.get("delta")
                        if isinstance(delta, str):
                            turn_transcript.append(delta)
                        continue

                    if event_type == "response.audio_transcript.done":
                        assert event.get("response_id") == response_id
                        transcript = event.get("transcript")
                        if isinstance(transcript, str):
                            turn_final_transcript = transcript
                        continue

                    if event_type == "response.done":
                        response = event.get("response")
                        done_response_id = event.get("response_id")
                        if not isinstance(done_response_id, str) and isinstance(response, dict):
                            done_response_id = response.get("id")
                        assert response_id is not None
                        assert done_response_id == response_id
                        if isinstance(response, dict) and isinstance(response.get("status"), str):
                            response_status = response["status"]
                        break

                    if event_type in {
                        "session.created",
                        "session.updated",
                        "rate_limits.updated",
                        "input_audio_buffer.speech_started",
                        "input_audio_buffer.speech_stopped",
                        "conversation.item.added",
                        "conversation.item.created",
                        "conversation.item.done",
                        "response.output_item.added",
                        "response.content_part.added",
                        "response.audio.done",
                        "response.content_part.done",
                        "response.output_item.done",
                    }:
                        continue

                    raise AssertionError(f"Unexpected Server VAD event: {event}")

                if event_type == "session.created":
                    continue

                if event_type == "input_audio_buffer.committed":
                    committed_events += 1
                    continue

                if event_type == "response.audio.delta":
                    delta_events += 1
                    sr = event.get("sample_rate_hz")
                    if isinstance(sr, int) and sr > 0:
                        output_sr = sr
                    audio_b64 = event.get("audio", "")
                    if audio_b64:
                        incremental.append(base64.b64decode(audio_b64))
                    continue

                if event_type == "transcription.delta":
                    d = event.get("delta", "")
                    if d:
                        text_chunks.append(d)
                    continue

                if event_type == "transcription.done":
                    final_text = event.get("text", "") or "".join(text_chunks)
                    continue

                if event_type == "response.audio.done":
                    break

                if event_type == "error":
                    raise AssertionError(f"WebSocket error: {event}")

                raise AssertionError(f"Unexpected WebSocket event: {event}")

            if server_vad:
                turn_text_value = (
                    turn_final_text or "".join(turn_text) or turn_final_transcript or "".join(turn_transcript)
                )
                turn_results.append(
                    {
                        "turn_index": turn_index,
                        "speech_started_item_id": speech_started_item_id,
                        "speech_stopped_item_id": speech_stopped_item_id,
                        "input_item_id": input_item_id,
                        "response_id": response_id,
                        "status": response_status,
                        "event_types": turn_event_types,
                        "output_pcm": b"".join(turn_audio),
                        "output_sample_rate": turn_output_sr,
                        "transcription_text": turn_text_value,
                        "delta_events": turn_delta_events,
                        "committed_events": turn_committed_events,
                    }
                )

    out_pcm = b"".join(incremental)
    return {
        "output_pcm": out_pcm,
        "output_sample_rate": output_sr,
        "transcription_text": final_text if final_text else "".join(text_chunks),
        "delta_events": delta_events,
        "committed_events": committed_events,
        "turns": turn_results,
    }


@pytest.fixture(scope="class")
def cached_silero_vad_artifact() -> str:
    """Prepare the pinned artifact before the serving subprocess starts."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=SILERO_VAD_REPO_ID,
        filename=SILERO_VAD_FILENAME,
        revision=SILERO_VAD_REVISION,
    )


def _synthetic_pcm16_input() -> bytes:
    syn = generate_synthetic_audio(
        10,
        1,
        sample_rate=16000,
        phrase_text=REALTIME_SYNTH_PHRASE_TEXT,
    )
    wav_bytes = base64.b64decode(syn["base64"])
    return _pcm16_mono_16k_from_wav_bytes(wav_bytes)


def _assert_realtime_smoke(result: dict) -> None:
    out_pcm = result["output_pcm"]
    assert result["delta_events"] >= 1
    assert out_pcm, "No output PCM from response.audio.delta"
    assert len(out_pcm) % 2 == 0
    assert len(out_pcm) >= 4096, "Output audio unexpectedly small"
    assert result["output_sample_rate"] > 0


def _assert_realtime_accuracy(
    result: dict,
    whisper_model_size: str = "large-v3",
    threshold: float = 0.8,
) -> None:
    """Assert that whisper transcription of audio output matches model text.

    Args:
        result: Roundtrip result dict from ``_run_realtime_audio_roundtrip``.
        whisper_model_size: Whisper model used to transcribe the generated audio
                   for the accuracy check. Defaults to ``large-v3``: the default
                   ``small`` model mishears short Chinese TTS clips (observed:
                   北京→韦京 and a dropped leading sentence, sim=0.443), which
                   caused spurious sim<0.8 failures under async_chunk codec
                   variability even though audio generation was correct. large-v3
                   transcribes these clips reliably, so a failure here now points
                   at the model, not the ASR grader.
        threshold: Minimum cosine similarity (with length penalty) required to
                   pass. Default 0.8. Do not lower per-callsite without data:
                   at 0.35 the assertion no longer detects real audio
                   regressions. If a variant genuinely needs a different gate
                   (e.g. whisper partial transcripts under async_chunk), propose
                   it in its own PR with measurements.
    """
    final_text = (result["transcription_text"] or "").strip()
    assert final_text, "Expected non-empty transcription (model text stream)"

    wav_out = _wav_bytes_from_pcm16(result["output_pcm"], result["output_sample_rate"])
    whisper_text = convert_audio_bytes_to_text(wav_out, model_size=whisper_model_size).strip()
    assert whisper_text, "Whisper returned empty string for synthesized output audio"

    sim = cosine_similarity_text(whisper_text.lower(), final_text.lower())
    assert sim > threshold, (
        f"Output audio transcript should match model text (sim={sim:.3f}, "
        f"threshold={threshold}): "
        f"whisper={whisper_text!r}, model_text={final_text!r}"
    )


class TestQwen3OmniRealtimeWebSocket:
    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    @pytest.mark.parametrize("omni_server", realtime_async_chunk_server_params, indirect=True)
    def test_streaming_audio_input_pcm_output_async_chunk(self, omni_server) -> None:
        """Merge CI: async_chunk on, paced upload, full accuracy check."""
        pcm16 = _synthetic_pcm16_input()

        result = asyncio.run(
            _run_realtime_audio_roundtrip(
                omni_server.host,
                omni_server.port,
                omni_server.model,
                pcm16,
                chunk_ms=100,
                send_delay_ms=SEND_DELAY_MS,
            )
        )

        _assert_realtime_smoke(result)
        _assert_realtime_accuracy(result)

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    @pytest.mark.parametrize("omni_server", realtime_sync_server_params, indirect=True)
    def test_server_vad_multi_turn_without_client_commit(
        self,
        cached_silero_vad_artifact: str,
        omni_server,
    ) -> None:
        """Two Qwen turns are endpointed without client commits."""
        assert cached_silero_vad_artifact
        pcm16 = _synthetic_pcm16_input()

        result = asyncio.run(
            _run_realtime_audio_roundtrip(
                omni_server.host,
                omni_server.port,
                omni_server.model,
                pcm16,
                chunk_ms=100,
                send_delay_ms=0,
                server_vad=True,
                turns=2,
            )
        )

        _assert_realtime_smoke(result)
        assert result["committed_events"] == 2
        turn_results = result["turns"]
        assert len(turn_results) == 2

        required_sequence = [
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "response.created",
            "response.audio.delta",
            "response.audio.done",
            "response.done",
        ]
        for turn in turn_results:
            assert turn["status"] == "completed"
            assert turn["committed_events"] == 1
            assert turn["delta_events"] >= 1
            assert turn["output_pcm"]
            assert turn["transcription_text"]
            event_types = turn["event_types"]
            positions = [event_types.index(event_type) for event_type in required_sequence]
            assert positions == sorted(positions)
            assert turn["speech_started_item_id"] == turn["speech_stopped_item_id"] == turn["input_item_id"]

        input_item_ids = [turn["input_item_id"] for turn in turn_results]
        response_ids = [turn["response_id"] for turn in turn_results]
        assert all(isinstance(value, str) and value for value in input_item_ids)
        assert all(isinstance(value, str) and value for value in response_ids)
        assert len(set(input_item_ids)) == 2
        assert len(set(response_ids)) == 2

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    @pytest.mark.parametrize("omni_server", realtime_sync_server_params, indirect=True)
    def test_streaming_audio_input_pcm_output(self, omni_server) -> None:
        """Merge CI: async_chunk off, no send delay, full accuracy check."""
        pcm16 = _synthetic_pcm16_input()

        result = asyncio.run(
            _run_realtime_audio_roundtrip(
                omni_server.host,
                omni_server.port,
                omni_server.model,
                pcm16,
                chunk_ms=100,
                send_delay_ms=0,
            )
        )

        _assert_realtime_smoke(result)
        _assert_realtime_accuracy(result)
