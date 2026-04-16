import logging
from pathlib import Path
from moviepy import VideoFileClip

logger = logging.getLogger(__name__)


def load_video(file_path: str) -> VideoFileClip:
    clip = VideoFileClip(str(file_path))
    logger.info(f"Loaded video: {Path(file_path).name}")
    return clip


def inspect_video(file_path: str) -> dict:
    clip = load_video(file_path)
    try:
        width, height = clip.size
        info = {
            "filename": Path(file_path).name,
            "duration_s": round(clip.duration, 2),
            "fps": clip.fps,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "has_audio": clip.audio is not None,
            "file_size_mb": round(Path(file_path).stat().st_size / (1024 * 1024), 2),
        }

        print(f"\n--- Video Properties: {Path(file_path).name} ---")
        for k, v in info.items():
            print(f"  {k:<20}: {v}")

        return info
    finally:
        clip.close()


def extract_audio_from_video(video_path: str, output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clip = load_video(video_path)

    try:
        if clip.audio is None:
            raise ValueError(f"Video has no audio track: {video_path}")

        clip.audio.write_audiofile(output_path, logger=None)
        size_kb = round(Path(output_path).stat().st_size / 1024, 1)
        logger.info(f"Extracted audio -> {output_path} ({size_kb} KB)")
        return output_path
    finally:
        clip.close()


if __name__ == "__main__":
    video_folder = Path("data/raw/video")
    video_files = list(video_folder.glob("*.mp4"))

    if not video_files:
        raise ValueError("No MP4 files found in data/raw/video")

    for file in video_files:
        print(f"\nProcessing: {file.name}")

        info = inspect_video(str(file))

        assert info["duration_s"] > 0
        assert info["fps"] > 0
        assert info["width"] > 0
        assert info["height"] > 0

        extract_audio_from_video(
            str(file),
            f"data/processed/audio/{file.stem}_extracted.mp3"
        )

        print(f"{file.name} passed.")

    print("\nAll video loader tests passed.")