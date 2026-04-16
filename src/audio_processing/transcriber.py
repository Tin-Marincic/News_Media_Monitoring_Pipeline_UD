import json
import logging
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel
from src.audio_processing.loader import load_audio

logger = logging.getLogger(__name__)

_model = None


def get_model(model_size: str = "base", device: str = "cpu") -> WhisperModel:
    global _model
    if _model is None:
        logger.info(f"Loading Whisper model: {model_size} on {device}")
        _model = WhisperModel(model_size, device=device, compute_type="int8")
        logger.info("Whisper model loaded.")
    return _model


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: str = None,
    word_timestamps: bool = True
) -> dict:
    model = get_model(model_size)
    logger.info(f"Transcribing: {Path(audio_path).name}")

    segments_gen, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=language,
        word_timestamps=word_timestamps,
    )
    segments = list(segments_gen)

    result = {
        "source_file": Path(audio_path).name,
        "source_path": str(audio_path),
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_s": round(info.duration, 2),
        "model": model_size,
        "segments": [],
        "full_text": "",
    }

    texts = []
    for seg in segments:
        seg_dict = {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "avg_log_prob": round(seg.avg_logprob, 4),
        }

        if word_timestamps and seg.words:
            seg_dict["words"] = [
                {
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.probability, 4),
                }
                for w in seg.words
            ]

        result["segments"].append(seg_dict)
        texts.append(seg.text.strip())

    result["full_text"] = " ".join(texts)

    logger.info(
        f"Transcription complete: {len(result['segments'])} segments, "
        f"language={info.language} ({info.language_probability:.0%})"
    )
    return result


def save_transcript_json(result: dict, output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved transcript JSON -> {output_path}")
    return output_path


def save_transcript_txt(result: dict, output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["full_text"])
    logger.info(f"Saved transcript TXT -> {output_path}")
    return output_path


def save_transcript_srt(result: dict, output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], 1):
            f.write(f"{i}\n")
            f.write(f"{fmt(seg['start'])} --> {fmt(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")

    logger.info(f"Saved SRT -> {output_path}")
    return output_path


def chunked_transcribe(
    audio_path: str,
    output_dir: str,
    model_size: str = "base",
    chunk_minutes: float = 5.0
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    audio = load_audio(audio_path)
    chunk_ms = int(chunk_minutes * 60 * 1000)
    total_ms = len(audio)
    n_chunks = (total_ms + chunk_ms - 1) // chunk_ms

    logger.info(
        f"Long audio: {total_ms / 1000:.0f}s -> {n_chunks} chunks of {chunk_minutes}min"
    )

    all_segments = []
    all_texts = []
    time_offset_s = 0.0

    detected_language = None
    detected_language_probability = None

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n_chunks):
            chunk_start = i * chunk_ms
            chunk_end = min((i + 1) * chunk_ms, total_ms)
            chunk_audio = audio[chunk_start:chunk_end]
            chunk_wav = Path(tmpdir) / f"chunk_{i:04d}.wav"
            chunk_audio.export(str(chunk_wav), format="wav")

            print(
                f"  Transcribing chunk {i + 1}/{n_chunks}: "
                f"{chunk_start / 1000:.0f}s - {chunk_end / 1000:.0f}s ..."
            )

            chunk_json = Path(output_dir) / f"chunk_{i:04d}.json"

            if chunk_json.exists():
                with open(chunk_json, "r", encoding="utf-8") as f:
                    chunk_result = json.load(f)
                print("    (loaded from cache)")
            else:
                chunk_result = transcribe_audio(
                    str(chunk_wav),
                    model_size=model_size,
                    word_timestamps=True
                )
                save_transcript_json(chunk_result, str(chunk_json))

            if detected_language is None:
                detected_language = chunk_result.get("language")
                detected_language_probability = chunk_result.get("language_probability")

            for seg in chunk_result["segments"]:
                adjusted = dict(seg)
                adjusted["start"] = round(seg["start"] + time_offset_s, 3)
                adjusted["end"] = round(seg["end"] + time_offset_s, 3)

                if "words" in adjusted:
                    adjusted["words"] = [
                        {
                            **w,
                            "start": round(w["start"] + time_offset_s, 3),
                            "end": round(w["end"] + time_offset_s, 3),
                        }
                        for w in adjusted["words"]
                    ]

                all_segments.append(adjusted)
                all_texts.append(seg["text"].strip())

            time_offset_s += (chunk_end - chunk_start) / 1000

    combined = {
        "source_file": Path(audio_path).name,
        "source_path": str(audio_path),
        "language": detected_language,
        "language_probability": detected_language_probability,
        "model": model_size,
        "total_chunks": n_chunks,
        "duration_s": round(total_ms / 1000, 2),
        "segments": all_segments,
        "full_text": " ".join(all_texts),
    }

    final_json = Path(output_dir) / "combined_transcript.json"
    final_txt = Path(output_dir) / "combined_transcript.txt"
    final_srt = Path(output_dir) / "combined_transcript.srt"

    save_transcript_json(combined, str(final_json))
    save_transcript_txt(combined, str(final_txt))
    save_transcript_srt(combined, str(final_srt))

    logger.info(f"Long transcription complete: {len(all_segments)} segments total")
    return combined


def transcribe_long_audio(
    audio_path: str,
    output_dir: str,
    model_size: str = "base",
    chunk_minutes: float = 5.0
) -> dict:
    return chunked_transcribe(
        audio_path=audio_path,
        output_dir=output_dir,
        model_size=model_size,
        chunk_minutes=chunk_minutes,
    )