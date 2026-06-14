from __future__ import annotations

import math

import numpy as np
import pytest

from src.asr.streaming_asr import (
    ChunkedASRSession,
    RollingAudioBuffer,
    calculate_rtf,
)


class FakeTranscribeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, audio, target_lang=None):
        self.calls.append({"audio": audio, "target_lang": target_lang})
        return ["partial text"]


def test_calculate_rtf() -> None:
    assert calculate_rtf(processing_time_s=0.25, audio_duration_s=1.0) == 0.25


def test_calculate_rtf_returns_inf_for_zero_audio_duration() -> None:
    assert math.isinf(calculate_rtf(processing_time_s=0.25, audio_duration_s=0.0))


def test_rolling_audio_buffer_keeps_most_recent_samples() -> None:
    buffer = RollingAudioBuffer(max_samples=5)

    buffer.append(np.array([1, 2, 3], dtype=np.float32))
    buffer.append(np.array([4, 5, 6], dtype=np.float32))

    np.testing.assert_array_equal(
        buffer.audio,
        np.array([2, 3, 4, 5, 6], dtype=np.float32),
    )
    assert buffer.sample_count == 5


def test_chunked_session_emits_event_with_rtf_and_metadata() -> None:
    clock_values = iter([10.0, 10.2])
    model = FakeTranscribeModel()
    session = ChunkedASRSession(
        model=model,
        sample_rate=10,
        target_lang="en-US",
        inference_interval_s=0.2,
        max_buffer_duration_s=1.0,
        clock=lambda: next(clock_values),
        timestamp_fn=lambda: "2026-01-01T00:00:00+00:00",
    )

    event = session.process_chunk(np.array([0.1, 0.2], dtype=np.float32))

    assert event is not None
    assert event["transcript"] == "partial text"
    assert event["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert event["latency_ms"] == pytest.approx(200.0)
    assert event["rtf"] == pytest.approx(1.0)
    assert event["audio_duration_s"] == 0.2
    assert event["mode"] == "chunked_fallback"
    assert event["target_lang"] == "en-US"
    assert event["language_readiness"] == "transcription_ready"
    assert model.calls[0]["target_lang"] == "en-US"


def test_chunked_session_waits_until_inference_interval() -> None:
    model = FakeTranscribeModel()
    session = ChunkedASRSession(
        model=model,
        sample_rate=10,
        inference_interval_s=0.3,
        max_buffer_duration_s=1.0,
    )

    event = session.process_chunk(np.array([0.1, 0.2], dtype=np.float32))

    assert event is None
    assert model.calls == []


def test_greek_session_labels_events_as_adaptation_ready() -> None:
    clock_values = iter([1.0, 1.1])
    session = ChunkedASRSession(
        model=FakeTranscribeModel(),
        sample_rate=10,
        target_lang="el-GR",
        inference_interval_s=0.1,
        clock=lambda: next(clock_values),
        timestamp_fn=lambda: "2026-01-01T00:00:00+00:00",
    )

    event = session.process_chunk(np.array([0.1], dtype=np.float32))

    assert event is not None
    assert event["language_readiness"] == "adaptation_ready"
    assert "exploratory" in event["warnings"][0]
