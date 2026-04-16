import logging
from pathlib import Path
from pydub import AudioSegment

from src.audio_processing.loader import load_audio

logger = logging.getLogger(__name__)


def trim_audio(audio: AudioSegment, start_ms: int, end_ms: int) -> AudioSegment:
    trimmed = audio[start_ms:end_ms]
    logger.info(
        f"Trimmed audio: {start_ms}ms -> {end_ms}ms "
        f"(was {len(audio)}ms, now {len(trimmed)}ms)"
    )
    return trimmed


def trim_silence(
    audio: AudioSegment,
    silence_start_ms: int = 2000,
    silence_end_ms: int = 2000
) -> AudioSegment:
    trimmed = audio[silence_start_ms: len(audio) - silence_end_ms]
    logger.info(
        f"Trimmed silence: start={silence_start_ms}ms, end={silence_end_ms}ms "
        f"(was {len(audio)}ms, now {len(trimmed)}ms)"
    )
    return trimmed


def concatenate_audio(clips: list[AudioSegment]) -> AudioSegment:
    if not clips:
        raise ValueError("Cannot concatenate empty list.")

    combined = clips[0]
    for clip in clips[1:]:
        combined = combined + clip

    logger.info(f"Concatenated {len(clips)} clips, total: {len(combined)}ms")
    return combined


def adjust_volume(audio: AudioSegment, db_change: float) -> AudioSegment:
    adjusted = audio + db_change
    logger.info(f"Volume adjusted by {db_change:+.1f} dB")
    return adjusted


def apply_fades(
    audio: AudioSegment,
    fade_in_ms: int = 1000,
    fade_out_ms: int = 2000
) -> AudioSegment:
    faded = audio.fade_in(fade_in_ms).fade_out(fade_out_ms)
    logger.info(f"Applied fade-in {fade_in_ms}ms / fade-out {fade_out_ms}ms")
    return faded


def export_audio(
    audio: AudioSegment,
    output_path: str,
    fmt: str = "mp3",
    bitrate: str = "192k"
) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if fmt in ("wav", "flac"):
        audio.export(str(output), format=fmt)
    else:
        audio.export(str(output), format=fmt, bitrate=bitrate)

    size_kb = round(output.stat().st_size / 1024, 1)
    logger.info(f"Exported audio -> {output} ({size_kb} KB)")
    return str(output)


def convert_audio_format(
    input_path: str,
    output_path: str,
    fmt: str = "mp3",
    bitrate: str = "192k"
) -> str:
    audio = load_audio(input_path)
    logger.info(f"Converting {input_path} -> {output_path} as {fmt.upper()}")
    return export_audio(audio, output_path, fmt=fmt, bitrate=bitrate)


if __name__ == "__main__":
    audio_folder = Path("data/raw/audio")
    audio_files = list(audio_folder.glob("*.mp3"))

    if not audio_files:
        raise ValueError("No MP3 files found in data/raw/audio")

    for file in audio_files:
        print(f"\nProcessing: {file.name}")

        audio = load_audio(str(file))
        print(f"Original duration: {len(audio) / 1000:.1f}s")

        trimmed = trim_audio(audio, start_ms=5000, end_ms=15000)
        assert len(trimmed) == 10000, f"Expected 10000ms, got {len(trimmed)}"
        print(f"Trimmed: {len(trimmed) / 1000:.1f}s (expected 10.0s)")

        doubled = concatenate_audio([trimmed, trimmed])
        assert len(doubled) == 20000, f"Expected 20000ms, got {len(doubled)}"
        print(f"Concatenated: {len(doubled) / 1000:.1f}s (expected 20.0s)")

        louder = adjust_volume(audio, +6)
        assert len(louder) == len(audio)
        print("Volume adjustment OK")

        softer = adjust_volume(audio, -4)
        assert len(softer) == len(audio)
        print("Negative volume adjustment OK")

        faded = apply_fades(audio, fade_in_ms=2000, fade_out_ms=3000)
        assert len(faded) == len(audio)
        print("Fade effects OK")

        export_audio(
            trimmed,
            f"data/processed/audio/{file.stem}_trimmed.wav",
            fmt="wav"
        )

        export_audio(
            trimmed,
            f"data/processed/audio/{file.stem}_trimmed.mp3",
            fmt="mp3",
            bitrate="192k"
        )

        export_audio(
            doubled,
            f"data/processed/audio/{file.stem}_concatenated.mp3",
            fmt="mp3",
            bitrate="192k"
        )

        export_audio(
            louder,
            f"data/processed/audio/{file.stem}_louder.mp3",
            fmt="mp3",
            bitrate="192k"
        )

        export_audio(
            softer,
            f"data/processed/audio/{file.stem}_softer.mp3",
            fmt="mp3",
            bitrate="192k"
        )

        export_audio(
            faded,
            f"data/processed/audio/{file.stem}_faded.mp3",
            fmt="mp3",
            bitrate="192k"
        )

        convert_audio_format(
            str(file),
            f"data/processed/audio/{file.stem}.flac",
            fmt="flac"
        )

        print(f"{file.name} passed.")

    print("\nAll processor tests passed.")