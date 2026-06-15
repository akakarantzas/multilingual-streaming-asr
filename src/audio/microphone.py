from __future__ import annotations

import queue
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 1600
_STOP_SENTINEL = object()


class MicrophoneAudioStream:
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        device: int | str | None = None,
        max_queue_chunks: int = 100,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device
        self._queue: queue.Queue[np.ndarray | object] = queue.Queue(maxsize=max_queue_chunks)
        self._stream: Any | None = None
        self._closed = True

    def __enter__(self) -> "MicrophoneAudioStream":
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is not installed. Install audio dependencies before using microphone capture."
            ) from exc

        self._closed = False
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.chunks()

    def chunks(self) -> Iterator[np.ndarray]:
        try:
            while True:
                if self._closed and self._queue.empty():
                    return
                item = self._queue.get()
                if item is _STOP_SENTINEL:
                    return
                yield item
        except KeyboardInterrupt:
            print("Microphone capture interrupted by user.", file=sys.stderr)
            self.close()
            return

    def close(self) -> None:
        self._closed = True
        self._wake_consumers()
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def _callback(self, indata: np.ndarray, _frames: int, _time_info: Any, status: Any) -> None:
        if status:
            print(f"Microphone stream status: {status}", file=sys.stderr)
        if self._closed:
            return

        chunk = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(chunk)

    def _wake_consumers(self) -> None:
        try:
            self._queue.put_nowait(_STOP_SENTINEL)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(_STOP_SENTINEL)


def microphone_chunks(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    device: int | str | None = None,
) -> Iterator[np.ndarray]:
    with MicrophoneAudioStream(
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        device=device,
    ) as stream:
        yield from stream.chunks()


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_SAMPLE_RATE",
    "MicrophoneAudioStream",
    "microphone_chunks",
]
