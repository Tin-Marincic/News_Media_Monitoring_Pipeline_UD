import logging
from pathlib import Path
from pydub import AudioSegment

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {
    '.wav': 'wav',
    '.mp3': 'mp3',
    '.flac': 'flac',
    '.ogg': 'ogg',
    '.aac': 'aac',
    '.m4a': 'mp4',
}


def load_audio(file_path: str) -> AudioSegment:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {ext}. Supported: {list(SUPPORTED_FORMATS.keys())}"
        )

    fmt = SUPPORTED_FORMATS[ext]
    audio = AudioSegment.from_file(str(path), format=fmt)
    logger.info(f"Loaded audio: {path.name} ({fmt.upper()})")
    return audio


def inspect_audio(file_path: str) -> dict:
    path = Path(file_path)
    audio = load_audio(file_path)

    if audio.channels == 1:
        channel_type = "Mono"
    elif audio.channels == 2:
        channel_type = "Stereo"
    else:
        channel_type = f"{audio.channels} channels"

    info = {
        "filename": path.name,
        "format": path.suffix.upper().lstrip('.'),
        "duration_ms": len(audio),
        "duration_sec": round(len(audio) / 1000, 2),
        "channels": audio.channels,
        "channel_type": channel_type,
        "frame_rate_hz": audio.frame_rate,
        "sample_width": audio.sample_width,
        "bit_depth": audio.sample_width * 8,
        "file_size_kb": round(path.stat().st_size / 1024, 1),
    }

    print(f"\n--- Audio Properties: {path.name} ---")
    for k, v in info.items():
        print(f"  {k:<20}: {v}")

    return info


def inspect_all_audio_files(audio_folder: str = "data/raw/audio") -> None:
    folder = Path(audio_folder)

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    found_files = False

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            found_files = True
            try:
                info = inspect_audio(str(file))

                assert info["duration_sec"] > 0
                assert info["channels"] >= 1
                assert info["frame_rate_hz"] > 0

                print(f"{file.name} passed.")
            except Exception as e:
                print(f"{file.name} failed: {e}")

    if not found_files:
        print("No supported audio files found.")

    print("Done checking all audio files.")


if __name__ == "__main__":
    inspect_all_audio_files()