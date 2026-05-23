from __future__ import annotations

from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}

MAX_VIDEO_MB = 700
MAX_AUDIO_MB = 150


def _size_mb(content: bytes) -> float:
    return len(content) / (1024 * 1024)


def validate_uploaded_file(
    filename: str,
    content: bytes,
    allowed_exts: set[str],
    max_size_mb: int,
) -> str | None:
    ext = Path(filename).suffix.lower()
    if ext not in allowed_exts:
        return f"Formato inválido: {ext or '(sem extensão)'}."

    size = _size_mb(content)
    if size > float(max_size_mb):
        return (
            f"Arquivo maior que o limite de {max_size_mb} MB "
            f"(tamanho atual: {size:.1f} MB)."
        )
    return None
