from pathlib import Path

from src.audio_processing.transcriber import (
    transcribe_audio,
    save_transcript_json,
    save_transcript_txt,
    save_transcript_srt,
    transcribe_long_audio,
)
from src.video_processing.loader import extract_audio_from_video
from src.storage.mongo import save_transcript


def main():
    # 1. SHORT AUDIO TEST
    short_audio = "data/raw/audio/news3.mp3"

    if Path(short_audio).exists():
        print(f"\nTesting short audio: {short_audio}")
        result = transcribe_audio(
            audio_path=short_audio,
            model_size="base",
            word_timestamps=True
        )

        print(f"Language detected : {result['language']} ({result['language_probability']:.0%} confidence)")
        print(f"Duration          : {result['duration_s']}s")
        print(f"Segments          : {len(result['segments'])}")
        print(f"Full text preview : {result['full_text'][:200]}...")

        print("\n--- Segments ---")
        for seg in result["segments"][:5]:
            print(f"  [{seg['start']:.1f}s -> {seg['end']:.1f}s] {seg['text']}")

        if result["segments"] and "words" in result["segments"][0]:
            print("\n--- Word Confidence (first segment) ---")
            for w in result["segments"][0]["words"]:
                conf_label = (
                    "HIGH" if w["probability"] >= 0.8
                    else "MED" if w["probability"] >= 0.5
                    else "LOW"
                )
                print(f"  [{conf_label}] {w['word']:<15} {w['probability']:.2f}")

        short_json = "data/processed/transcripts/news3.json"
        short_txt = "data/processed/transcripts/news3.txt"
        short_srt = "data/processed/transcripts/news3.srt"

        save_transcript_json(result, short_json)
        save_transcript_txt(result, short_txt)
        save_transcript_srt(result, short_srt)

        save_transcript(
            transcript_result=result,
            source_path=short_audio,
            source_type="audio",
            json_path=short_json,
            txt_path=short_txt,
            srt_path=short_srt,
        )

        print("\nSaved: JSON, TXT, SRT + MongoDB transcript record")

    # 2. VIDEO AUDIO TEST
    video_path = "data/raw/video/news1.mp4"
    audio_out = "data/processed/audio/news1_from_video.mp3"
    trans_out_json = "data/processed/transcripts/news1_video_audio.json"
    trans_out_txt = "data/processed/transcripts/news1_video_audio.txt"
    trans_out_srt = "data/processed/transcripts/news1_video_audio.srt"

    if Path(video_path).exists():
        print("\nStep 1: Extracting audio from video...")
        extracted_audio = extract_audio_from_video(video_path, audio_out)
        print(f"Audio saved: {extracted_audio}")

        print("Step 2: Transcribing extracted audio...")
        result = transcribe_audio(extracted_audio, model_size="base", word_timestamps=True)

        print(f"Language detected : {result['language']} ({result['language_probability']:.0%} confidence)")
        print(f"Duration          : {result['duration_s']}s")
        print(f"Segments          : {len(result['segments'])}")
        print(f"Text preview      : {result['full_text'][:300]}...")

        save_transcript_json(result, trans_out_json)
        save_transcript_txt(result, trans_out_txt)
        save_transcript_srt(result, trans_out_srt)

        save_transcript(
            transcript_result=result,
            source_path=extracted_audio,
            source_type="video_audio",
            json_path=trans_out_json,
            txt_path=trans_out_txt,
            srt_path=trans_out_srt,
        )

        print("Saved extracted video audio transcript: JSON, TXT, SRT + MongoDB transcript record")

    # 3. LONG AUDIO TEST
    long_audio = "data/raw/audio/news1.mp3"
    long_output_dir = "data/processed/transcripts/news1_chunks"

    if Path(long_audio).exists():
        print("\nTesting long audio with chunking...")
        result = transcribe_long_audio(
            audio_path=long_audio,
            output_dir=long_output_dir,
            model_size="base",
            chunk_minutes=5.0
        )

        print(f"Language          : {result.get('language')} ({result.get('language_probability')})")
        print(f"Total segments    : {len(result['segments'])}")
        print(f"Full text length  : {len(result['full_text'])} characters")
        print(f"Preview           : {result['full_text'][:400]}...")

        long_json = f"{long_output_dir}/combined_transcript.json"
        long_txt = f"{long_output_dir}/combined_transcript.txt"
        long_srt = f"{long_output_dir}/combined_transcript.srt"

        save_transcript(
            transcript_result=result,
            source_path=long_audio,
            source_type="audio",
            json_path=long_json,
            txt_path=long_txt,
            srt_path=long_srt,
        )

        print("Saved long audio chunked transcript record to MongoDB")


if __name__ == "__main__":
    main()