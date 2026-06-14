from __future__ import annotations

import inspect
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from src.asr.infer import (
    ALLOWED_TARGET_LANGS,
    _build_transcribe_call,
    extract_transcript_text,
)


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_INFERENCE_INTERVAL_S = 1.0
DEFAULT_MAX_BUFFER_DURATION_S = 10.0


class RollingAudioBuffer:
    def __init__(self, max_samples: int) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.max_samples = max_samples
        self._samples = np.empty(0, dtype=np.float32)

    def append(self, chunk: np.ndarray) -> np.ndarray:
        chunk_array = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk_array.size == 0:
            return self.audio
        self._samples = np.concatenate((self._samples, chunk_array))
        if self._samples.size > self.max_samples:
            self._samples = self._samples[-self.max_samples :]
        return self.audio

    @property
    def audio(self) -> np.ndarray:
        return self._samples.copy()

    @property
    def sample_count(self) -> int:
        return int(self._samples.size)

    def duration_s(self, sample_rate: int) -> float:
        return self.sample_count / sample_rate


class ChunkedASRSession:
    def __init__(
        self,
        model: Any,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        target_lang: str = "en-US",
        inference_interval_s: float = DEFAULT_INFERENCE_INTERVAL_S,
        max_buffer_duration_s: float = DEFAULT_MAX_BUFFER_DURATION_S,
        clock: Callable[[], float] = time.perf_counter,
        timestamp_fn: Callable[[], str] | None = None,
    ) -> None:
        if target_lang not in ALLOWED_TARGET_LANGS:
            allowed = ", ".join(sorted(ALLOWED_TARGET_LANGS))
            raise ValueError(f"Unsupported target_lang {target_lang!r}. Allowed values: {allowed}")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if inference_interval_s <= 0:
            raise ValueError("inference_interval_s must be positive")
        if max_buffer_duration_s <= 0:
            raise ValueError("max_buffer_duration_s must be positive")

        self.model = model
        self.sample_rate = sample_rate
        self.target_lang = target_lang
        self.inference_interval_s = inference_interval_s
        self.inference_interval_samples = max(1, int(sample_rate * inference_interval_s))
        self.buffer = RollingAudioBuffer(max_samples=int(sample_rate * max_buffer_duration_s))
        self.clock = clock
        self.timestamp_fn = timestamp_fn or _utc_timestamp
        self.mode = _detect_native_streaming_mode(model)
        self.language_readiness, self.session_warnings = _language_readiness(target_lang)
        self._samples_since_inference = 0
        self._event_count = 0

    def process_chunk(self, chunk: np.ndarray) -> dict | None:
        chunk_array = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk_array.size == 0:
            return None

        self.buffer.append(chunk_array)
        self._samples_since_inference += int(chunk_array.size)
        if self._samples_since_inference < self.inference_interval_samples:
            return None

        self._samples_since_inference = 0
        return self._run_inference_event()

    def process_chunks(self, chunks: Iterable[np.ndarray]) -> Iterator[dict]:
        for chunk in chunks:
            event = self.process_chunk(chunk)
            if event is not None:
                yield event

    def _run_inference_event(self) -> dict:
        audio = self.buffer.audio
        audio_duration_s = self.buffer.duration_s(self.sample_rate)
        warnings = list(self.session_warnings)

        start = self.clock()
        transcript = self._run_native_streaming(audio, warnings)
        if transcript is None:
            transcript = self._run_chunked_fallback(audio, warnings)
        processing_time_s = self.clock() - start

        event = {
            "transcript": transcript,
            "timestamp": self.timestamp_fn(),
            "latency_ms": processing_time_s * 1000,
            "rtf": calculate_rtf(processing_time_s, audio_duration_s),
            "audio_duration_s": audio_duration_s,
            "mode": self.mode,
            "target_lang": self.target_lang,
            "language_readiness": self.language_readiness,
            "warnings": warnings,
        }
        print(
            f"[{event['timestamp']}] partial transcript: {event['transcript']} "
            f"(latency_ms={event['latency_ms']:.2f}, rtf={event['rtf']:.3f}, mode={event['mode']})"
        )
        return event

    def _run_native_streaming(self, audio: np.ndarray, warnings: list[str]) -> str | None:
        if self.mode != "native_streaming":
            return None

        streaming_method = _native_streaming_method(self.model)
        if streaming_method is None:
            warnings.append(
                "Native streaming mode was requested but no callable native streaming method was found."
            )
            self.mode = "chunked_fallback"
            return None

        kwargs: dict[str, Any] = {}
        if _call_accepts_parameter(streaming_method, "target_lang"):
            kwargs["target_lang"] = self.target_lang
        if _call_accepts_parameter(streaming_method, "sample_rate"):
            kwargs["sample_rate"] = self.sample_rate
        return extract_transcript_text(streaming_method(audio, **kwargs))

    def _run_chunked_fallback(self, audio: np.ndarray, warnings: list[str]) -> str:
        transcribe = getattr(self.model, "transcribe", None)
        if not callable(transcribe):
            raise RuntimeError(
                "Loaded ASR model does not expose a documented native streaming method or "
                "model.transcribe() for chunked fallback. Check the installed NeMo/Nemotron "
                "3.5 ASR streaming API before running microphone ASR."
            )

        if _call_accepts_parameter(transcribe, "audio"):
            kwargs: dict[str, Any] = {"audio": [audio]}
            if _call_accepts_parameter(transcribe, "sample_rate"):
                kwargs["sample_rate"] = self.sample_rate
            if _call_accepts_parameter(transcribe, "target_lang"):
                kwargs["target_lang"] = self.target_lang
            else:
                warnings.append(
                    "Installed model.transcribe() API does not expose target_lang; continuing "
                    "without target-language conditioning."
                )
            return extract_transcript_text(transcribe(**kwargs))

        return self._run_chunked_fallback_with_temp_wav(audio, transcribe, warnings)

    def _run_chunked_fallback_with_temp_wav(
        self,
        audio: np.ndarray,
        transcribe: Any,
        warnings: list[str],
    ) -> str:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "Chunked fallback needs soundfile to create temporary WAV buffers when "
                "model.transcribe() does not accept in-memory audio."
            ) from exc

        self._event_count += 1
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=f"_stream_chunk_{self._event_count}.wav",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
            sf.write(temp_path, audio, self.sample_rate)
            transcribe_call, _target_lang_support = _build_transcribe_call(
                transcribe=transcribe,
                audio_file=temp_path,
                target_lang=self.target_lang,
                warnings=warnings,
            )
            return extract_transcript_text(
                transcribe(*transcribe_call.args, **transcribe_call.kwargs)
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def calculate_rtf(processing_time_s: float, audio_duration_s: float) -> float:
    if audio_duration_s <= 0:
        return float("inf")
    return processing_time_s / audio_duration_s


def _detect_native_streaming_mode(model: Any) -> str:
    if _native_streaming_method(model) is not None:
        return "native_streaming"
    return "chunked_fallback"


def _native_streaming_method(model: Any) -> Any | None:
    # Conservative detection only. Do not claim native streaming unless the loaded
    # model exposes a clearly named streaming/cache-aware method.
    for method_name in (
        "transcribe_streaming",
        "streaming_transcribe",
        "transcribe_stream",
    ):
        method = getattr(model, method_name, None)
        if callable(method):
            return method
    return None


def _language_readiness(target_lang: str) -> tuple[str, list[str]]:
    if target_lang in {"en-US", "en-GB"}:
        return "transcription_ready", []
    if target_lang == "el-GR":
        return (
            "adaptation_ready",
            [
                "Greek el-GR streaming output is exploratory/adaptation-ready until "
                "fine-tuning/evaluation is measured."
            ],
        )
    return "auto", []


def _call_accepts_parameter(callable_obj: Any, parameter_name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False

    if parameter_name in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ChunkedASRSession",
    "RollingAudioBuffer",
    "calculate_rtf",
]
