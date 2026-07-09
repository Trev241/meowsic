from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .errors import AudioFormatError
from .types import AudioBuffer


def load_wav(path: str | Path) -> AudioBuffer:
    """Load a PCM WAV file as normalized float32 audio."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            sample_rate = reader.getframerate()
            sample_width = reader.getsampwidth()
            frame_count = reader.getnframes()
            raw = reader.readframes(frame_count)
    except wave.Error as exc:
        raise AudioFormatError(f"Unsupported or invalid WAV file: {path}") from exc

    if channels <= 0 or frame_count <= 0:
        raise AudioFormatError(f"Invalid or empty WAV file: {path}")

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        data = _decode_24bit(raw).astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioFormatError(f"Unsupported WAV sample width {sample_width} bytes: {path}")

    return AudioBuffer(data.reshape(-1, channels), sample_rate)


def write_wav(path: str | Path, audio: AudioBuffer, *, overwrite: bool = False) -> Path:
    """Write normalized float audio to a 16-bit PCM WAV file."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio.samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(audio.channels)
        writer.setsampwidth(2)
        writer.setframerate(audio.sample_rate)
        writer.writeframes(pcm.tobytes())
    return path


def _decode_24bit(raw: bytes) -> np.ndarray:
    bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    values = (
        bytes_[:, 0].astype(np.int32)
        | (bytes_[:, 1].astype(np.int32) << 8)
        | (bytes_[:, 2].astype(np.int32) << 16)
    )
    sign_bit = 1 << 23
    return (values ^ sign_bit) - sign_bit
