import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.logger import logging

from src.api.client import fetch_news
from src.parsing.parsers import (
    parse_json_files,
    extract_text_from_pdf,
    extract_text_from_two_column_pdf,
    extract_text_from_word,
    extract_text_from_two_column_word,
    extract_word_runs,
    extract_data_from_excel,
    extract_summary_from_excel,
    read_file_with_encoding,
)
from src.storage.mongo import (
    save_to_mongo,
    save_batch_results_to_mongo,
    save_transcript,
)

from src.scraping.scraper import scrape_hockey_teams, scrape_hockey_teams_multi_page
from src.scraping.dynamic_scraper import scrape_ajax_movies_api
from src.ocr.ocr_utils import ocr_image, ocr_scanned_pdf

from src.image_processing.downloader import load_articles_from_json, download_article_images
from src.image_processing.batch import batch_process_images

from src.audio_processing.loader import load_audio
from src.audio_processing.processor import trim_audio, apply_fades, export_audio
from src.audio_processing.transcriber import (
    transcribe_audio,
    transcribe_long_audio,
    save_transcript_json,
    save_transcript_txt,
    save_transcript_srt,
)
from src.video_processing.loader import inspect_video, extract_audio_from_video
from src.video_processing.frame_extractor import extract_keyframes

from src.analytics.numpy_ops import (
    demonstrate_array_creation,
    print_array_info,
    vectorized_operations,
    axis_reductions,
    broadcasting_example,
)
from src.analytics.data_loader import (
    load_from_mongodb,
    save_to_csv,
    load_from_csv,
    chunked_stats,
    process_chunks_per_language,
    optimise_dtypes,
    memory_comparison,
)
from src.analytics.explorer import (
    inspect_shape,
    print_info,
    describe_numeric,
    describe_all,
    value_counts_report,
    nunique_report,
    extract_release_year,
    plot_distributions,
)
from src.analytics.selector import (
    select_columns,
    loc_filter,
    iloc_sample,
    boolean_filter,
    isin_filter,
    between_filter,
)
from src.analytics.regex_ops import (
    extract_year_from_title,
    extract_any_year_from_title,
    filter_titles_starting_with,
    extract_number_from_title,
    crime_overview_count,
    crime_overview_rows,
    short_overviews,
    extract_genres,
    top_genres,
    validate_movie_ids,
)
from src.analytics.quality_report import (
    missing_value_report,
    zero_as_missing,
    outlier_report,
    rating_validity_report,
    duplicate_id_report,
    title_quality_report,
    format_consistency_report,
    full_quality_report,
    save_quality_report,
    save_missing_heatmap,
)

from src.cleaning.clean_pipeline import run_cleaning_pipeline_from_csv


from src.analytics.db_connector import setup_mysql_from_cleaned_data
from src.analytics.data_combiner import (
    prepare_metadata_df,
    merge_metadata_with_metrics,
    compare_join_types,
    save_join_count_chart,
    save_combined_outputs,
)
from src.analytics.pivot_builder import (
    add_primary_category,
    add_analysis_year,
    wide_to_long_metrics,
    long_to_wide_metrics,
    build_category_year_pivot,
    build_language_decade_crosstab,
    save_pivot_outputs,
)
from src.analytics.aggregator import (
    category_summary,
    document_type_summary,
    yearly_trends,
    top_n_per_group,
    add_group_average_columns,
    filter_large_categories,
    save_yearly_trends_chart,
    save_aggregation_outputs,
)
from src.analytics.time_series import (
    parse_news_dates,
    add_date_components,
    build_monthly_time_series,
    resample_yearly,
    resample_quarterly,
    add_rolling_averages,
    save_time_series_chart,
    save_time_series_outputs,
)
from src.analytics.mongo_pipeline import (
    run_category_aggregation,
    run_document_type_aggregation,
    run_source_aggregation,
    save_mongo_aggregation_outputs,
)
from src.analytics.insight_reporter import (
    generate_insight_questions,
    save_insight_outputs,
)


def save_standard_transcript_outputs(result: dict, base_output_path: str) -> tuple[str, str, str]:
    """
    Save transcript as JSON, TXT, and SRT using a shared base path.
    Example base_output_path: data/processed/transcripts/news3
    """
    json_path = f"{base_output_path}.json"
    txt_path = f"{base_output_path}.txt"
    srt_path = f"{base_output_path}.srt"

    save_transcript_json(result, json_path)
    save_transcript_txt(result, txt_path)
    save_transcript_srt(result, srt_path)

    return json_path, txt_path, srt_path


def run_audio_video_stage():
    logging.info("=== Audio/Video Processing Stage Started ===")

    raw_audio_dir = Path("data/raw/audio")
    raw_video_dir = Path("data/raw/video")
    processed_audio_dir = Path("data/processed/audio")
    processed_frames_dir = Path("data/processed/frames")
    processed_transcripts_dir = Path("data/processed/transcripts")

    processed_audio_dir.mkdir(parents=True, exist_ok=True)
    processed_frames_dir.mkdir(parents=True, exist_ok=True)
    processed_transcripts_dir.mkdir(parents=True, exist_ok=True)

    if raw_audio_dir.exists():
        for audio_file in raw_audio_dir.glob("*.mp3"):
            try:
                logging.info(f"Processing audio file: {audio_file.name}")

                audio = load_audio(str(audio_file))
                logging.info(
                    f"Loaded audio {audio_file.name}: duration={len(audio) / 1000:.2f}s, "
                    f"channels={audio.channels}, frame_rate={audio.frame_rate}"
                )

                trimmed = trim_audio(audio, 0, min(30000, len(audio)))
                faded = apply_fades(trimmed, fade_in_ms=1000, fade_out_ms=2000)

                processed_clip_path = processed_audio_dir / f"{audio_file.stem}_clip.mp3"
                export_audio(faded, str(processed_clip_path), fmt="mp3", bitrate="192k")
                logging.info(f"Saved processed audio clip: {processed_clip_path}")

                if len(audio) > 5 * 60 * 1000:
                    logging.info(f"Using chunked transcription for long audio: {audio_file.name}")
                    chunk_output_dir = processed_transcripts_dir / f"{audio_file.stem}_chunks"

                    result = transcribe_long_audio(
                        audio_path=str(audio_file),
                        output_dir=str(chunk_output_dir),
                        model_size="base",
                        chunk_minutes=5.0,
                    )

                    json_path = str(chunk_output_dir / "combined_transcript.json")
                    txt_path = str(chunk_output_dir / "combined_transcript.txt")
                    srt_path = str(chunk_output_dir / "combined_transcript.srt")
                else:
                    logging.info(f"Using standard transcription for audio: {audio_file.name}")
                    result = transcribe_audio(
                        audio_path=str(audio_file),
                        model_size="base",
                        word_timestamps=True,
                    )

                    base_output = processed_transcripts_dir / audio_file.stem
                    json_path, txt_path, srt_path = save_standard_transcript_outputs(
                        result,
                        str(base_output),
                    )

                save_transcript(
                    transcript_result=result,
                    source_path=str(audio_file),
                    source_type="audio",
                    json_path=json_path,
                    txt_path=txt_path,
                    srt_path=srt_path,
                )

                logging.info(f"Finished audio stage for: {audio_file.name}")

            except Exception as e:
                logging.error(f"Audio processing failed for {audio_file.name}: {e}")

    if raw_video_dir.exists():
        for video_file in raw_video_dir.glob("*.mp4"):
            try:
                logging.info(f"Processing video file: {video_file.name}")

                video_info = inspect_video(str(video_file))
                logging.info(
                    f"Video info for {video_file.name}: "
                    f"duration={video_info['duration_s']}s, "
                    f"fps={video_info['fps']}, "
                    f"resolution={video_info['resolution']}"
                )

                video_frames_dir = processed_frames_dir / video_file.stem
                frames = extract_keyframes(
                    str(video_file),
                    str(video_frames_dir),
                    interval_seconds=10.0,
                )
                logging.info(f"Extracted {len(frames)} keyframes for {video_file.name}")

                extracted_audio_path = processed_audio_dir / f"{video_file.stem}_from_video.mp3"
                extract_audio_from_video(str(video_file), str(extracted_audio_path))
                logging.info(f"Extracted audio from video: {extracted_audio_path}")

                result = transcribe_audio(
                    str(extracted_audio_path),
                    model_size="base",
                    word_timestamps=True,
                )

                base_output = processed_transcripts_dir / f"{video_file.stem}_video_audio"
                json_path, txt_path, srt_path = save_standard_transcript_outputs(
                    result,
                    str(base_output),
                )

                save_transcript(
                    transcript_result=result,
                    source_path=str(extracted_audio_path),
                    source_type="video_audio",
                    json_path=json_path,
                    txt_path=txt_path,
                    srt_path=srt_path,
                )

                logging.info(f"Finished video stage for: {video_file.name}")

            except Exception as e:
                logging.error(f"Video processing failed for {video_file.name}: {e}")

    logging.info("=== Audio/Video Processing Stage Complete ===")


def upload_lab8_charts_to_drive(chart_paths: list[str]) -> list[dict]:
    """
    Optional Google Drive upload for Lab 8 charts.
    """
    logging.info("=== Lab 8 Google Drive Chart Upload Started ===")

    upload_results = []

    if not chart_paths:
        logging.warning("No Lab 8 charts available for upload")
        return upload_results

    try:
        from googleapiclient.http import MediaFileUpload
        from src.utils.upload_utils import authenticate_drive, FOLDER_ID
    except Exception as e:
        logging.warning(f"Google Drive upload skipped because upload utilities are unavailable: {e}")
        return upload_results

    amila_email = os.getenv("AMILA_EMAIL")

    try:
        service = authenticate_drive()
    except Exception as e:
        logging.error(f"Google Drive authentication failed: {e}")
        return upload_results

    for chart_path in chart_paths:
        try:
            path = Path(chart_path)

            if not path.exists():
                logging.warning(f"Chart path does not exist, skipping upload: {chart_path}")
                continue

            file_metadata = {"name": path.name}

            if FOLDER_ID:
                file_metadata["parents"] = [FOLDER_ID]

            media = MediaFileUpload(
                str(path),
                mimetype="image/png",
                resumable=False,
            )

            uploaded_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, webViewLink",
            ).execute()

            file_id = uploaded_file.get("id")
            file_url = uploaded_file.get("webViewLink")

            if amila_email and file_id:
                service.permissions().create(
                    fileId=file_id,
                    body={
                        "type": "user",
                        "role": "reader",
                        "emailAddress": amila_email,
                    },
                    sendNotificationEmail=False,
                ).execute()

                logging.info(f"Shared chart with Amila: {path.name} -> {amila_email}")

            upload_results.append({
                "file_name": path.name,
                "local_path": str(path),
                "drive_file_id": file_id,
                "drive_url": file_url,
                "shared_with": amila_email if amila_email else "",
            })

            logging.info(f"Uploaded Lab 8 chart to Google Drive: {path.name}")

        except Exception as e:
            logging.error(f"Failed to upload chart {chart_path}: {e}")

    logging.info("=== Lab 8 Google Drive Chart Upload Complete ===")

    return upload_results


def run_lab8_analytics_stage():
    """
    Run Lab 8 analytics on the integrated News Media Monitoring Pipeline
    dataset stored in MongoDB.
    """
    logging.info("=== Lab 8 News Analytics Stage Started ===")

    analytics_dir = Path("data/processed/analytics")
    charts_dir = analytics_dir / "charts"
    reports_dir = analytics_dir / "reports"

    analytics_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = analytics_dir / "raw_news_data.csv"
    optimized_csv_path = analytics_dir / "optimized_news_data.csv"

    logging.info("Lab 8 Part 2: NumPy operations started")

    arrays = demonstrate_array_creation()
    print_array_info(arrays)

    numpy_array_rows = []

    for name, arr in arrays.items():
        numpy_array_rows.append({
            "array_name": name,
            "shape": str(arr.shape),
            "dtype": str(arr.dtype),
            "ndim": arr.ndim,
            "size": arr.size,
            "itemsize": arr.itemsize,
        })

        logging.info(
            "NumPy array %s: shape=%s dtype=%s ndim=%s size=%s itemsize=%s",
            name,
            arr.shape,
            arr.dtype,
            arr.ndim,
            arr.size,
            arr.itemsize,
        )

    pd.DataFrame(numpy_array_rows).to_csv(
        reports_dir / "numpy_array_info.csv",
        index=False,
    )

    rating_score = np.array([8.2, 7.4, 6.9, 9.1, 5.8, 8.7])
    mentions = np.array([34, 28, 19, 23, 14, 17])

    vectorized_result = vectorized_operations(rating_score, mentions)

    vectorized_output = {
        "rating_score": rating_score,
        "mentions": mentions,
        "normalised": vectorized_result["normalised"],
        "weighted": vectorized_result["weighted"],
        "high_rated": vectorized_result["high_rated"],
        "broadcasting_normalised": broadcasting_example(rating_score),
    }

    if "high_impact" in vectorized_result:
        vectorized_output["high_impact"] = vectorized_result["high_impact"]

    if "quality" in vectorized_result:
        vectorized_output["quality"] = vectorized_result["quality"]

    pd.DataFrame(vectorized_output).to_csv(
        reports_dir / "numpy_vectorized_operations.csv",
        index=False,
    )

    news_matrix = np.array([
        [8.2, 34, 120.4],
        [7.4, 28, 95.7],
        [6.9, 19, 80.2],
        [9.1, 23, 63.1],
    ])

    axis_result = axis_reductions(news_matrix)

    pd.DataFrame({
        "col_means": axis_result["col_means"],
        "col_stds": axis_result["col_stds"],
    }).to_csv(
        reports_dir / "numpy_axis_reductions.csv",
        index=False,
    )

    pd.DataFrame({
        "row_means": axis_result["row_means"],
    }).to_csv(
        reports_dir / "numpy_row_reductions.csv",
        index=False,
    )

    logging.info("Lab 8 Part 2: NumPy operations complete")

    logging.info("Lab 8 Part 3: Loading and memory management started")

    news_df = load_from_mongodb()

    if news_df.empty:
        logging.warning("Lab 8 analytics skipped because MongoDB returned no news records")
        return

    save_to_csv(news_df, str(raw_csv_path))

    csv_df = load_from_csv(str(raw_csv_path))

    chunk_results = chunked_stats(
        str(raw_csv_path),
        chunk_size=50,
        rating_column="rating_score",
        language_column="language",
    )

    pd.DataFrame([{
        "global_mean_rating_score": chunk_results["global_mean"],
        "total_rows": chunk_results["total_rows"],
        "rating_count": chunk_results["rating_count"],
    }]).to_csv(
        reports_dir / "chunked_global_rating_score_mean.csv",
        index=False,
    )

    chunk_results["language_df"].to_csv(
        reports_dir / "chunked_language_rating_score_stats.csv",
        index=False,
    )

    language_stats = process_chunks_per_language(
        str(raw_csv_path),
        chunk_size=50,
        rating_column="rating_score",
        language_column="language",
    )

    language_stats.to_csv(
        reports_dir / "per_language_accumulators.csv",
        index=False,
    )

    optimized_df = optimise_dtypes(csv_df)

    memory_stats = memory_comparison(csv_df, optimized_df)

    pd.DataFrame([memory_stats]).to_csv(
        reports_dir / "memory_optimisation_report.csv",
        index=False,
    )

    save_to_csv(optimized_df, str(optimized_csv_path))

    logging.info("Lab 8 Part 3: Loading and memory management complete")

    logging.info("Lab 8 Part 4: EDA started")

    eda_df = extract_release_year(csv_df)

    shape_info = inspect_shape(eda_df)

    pd.DataFrame([shape_info]).to_csv(
        reports_dir / "eda_shape_report.csv",
        index=False,
    )

    print_info(eda_df)

    describe_numeric(eda_df).to_csv(
        reports_dir / "eda_numeric_describe.csv",
    )

    describe_all(eda_df).to_csv(
        reports_dir / "eda_full_describe.csv",
    )

    unique_report = nunique_report(eda_df)

    unique_report.to_csv(
        reports_dir / "eda_nunique_report.csv",
        index=False,
    )

    counts_report = value_counts_report(eda_df)

    for col, report in counts_report.items():
        report["counts"].to_csv(
            reports_dir / f"eda_value_counts_{col}.csv",
            header=["count"],
        )

    saved_charts = plot_distributions(
        eda_df,
        output_dir=str(charts_dir),
    )

    logging.info("Lab 8 Part 4: EDA complete")

    logging.info("Lab 8 Part 5: Selection and filtering started")

    select_columns(
        eda_df,
        [
            "record_id",
            "title",
            "document_type",
            "category",
            "rating_score",
            "mentions",
            "popularity",
            "language",
            "published_year",
        ],
    ).to_csv(
        reports_dir / "selection_selected_columns.csv",
        index=False,
    )

    loc_filter(
        eda_df,
        min_rating_score=5.0,
    ).to_csv(
        reports_dir / "selection_loc_filter.csv",
        index=False,
    )

    iloc_sample(
        eda_df,
        step=10,
    ).to_csv(
        reports_dir / "selection_iloc_sample.csv",
        index=False,
    )

    boolean_filter(
        eda_df,
        min_rating_score=5.0,
        min_mentions=1,
        min_popularity=10.0,
    ).to_csv(
        reports_dir / "selection_boolean_filter.csv",
        index=False,
    )

    isin_filter(
        eda_df,
        values=["news_api", "json", "pdf", "word", "excel"],
        column="document_type",
        exclude=False,
    ).to_csv(
        reports_dir / "selection_isin_filter.csv",
        index=False,
    )

    isin_filter(
        eda_df,
        values=["news_api"],
        column="document_type",
        exclude=True,
    ).to_csv(
        reports_dir / "selection_isin_exclusion_filter.csv",
        index=False,
    )

    between_filter(
        eda_df,
        col="rating_score",
        low=2.0,
        high=8.0,
    ).to_csv(
        reports_dir / "selection_between_filter.csv",
        index=False,
    )

    logging.info("Lab 8 Part 5: Selection and filtering complete")

    logging.info("Lab 8 Part 6: Regex operations started")

    regex_df = eda_df.copy()

    if "title" in regex_df.columns:
        regex_df["title_year_parentheses"] = extract_year_from_title(regex_df["title"])
        regex_df["title_any_year"] = extract_any_year_from_title(regex_df["title"])

    titles_starting_with_the = filter_titles_starting_with(
        regex_df,
        prefix="The",
    )

    titles_starting_with_the.to_csv(
        reports_dir / "regex_titles_starting_with_the.csv",
        index=False,
    )

    regex_df = extract_number_from_title(regex_df)

    crime_count = crime_overview_count(regex_df)

    pd.DataFrame([{
        "crime_related_content_count": crime_count,
    }]).to_csv(
        reports_dir / "regex_crime_content_count.csv",
        index=False,
    )

    crime_overview_rows(regex_df).to_csv(
        reports_dir / "regex_crime_content_rows.csv",
        index=False,
    )

    short_overviews(
        regex_df,
        max_chars=40,
    ).to_csv(
        reports_dir / "regex_short_content_rows.csv",
        index=False,
    )

    regex_df = extract_genres(regex_df)

    category_counts = top_genres(regex_df, n=15)

    pd.DataFrame(
        category_counts,
        columns=["category_label", "count"],
    ).to_csv(
        reports_dir / "regex_top_categories.csv",
        index=False,
    )

    regex_df = validate_movie_ids(regex_df)

    regex_df.to_csv(
        reports_dir / "regex_processed_news_dataset.csv",
        index=False,
    )

    logging.info("Lab 8 Part 6: Regex operations complete")

    logging.info("Lab 8 Part 7: Data quality reporting started")

    missing_report = missing_value_report(eda_df)

    missing_report.to_csv(
        reports_dir / "quality_missing_value_report.csv",
        index=False,
    )

    zero_report = zero_as_missing(eda_df)

    zero_report.to_csv(
        reports_dir / "quality_zero_as_missing_report.csv",
        index=False,
    )

    outlier_report(eda_df).to_csv(
        reports_dir / "quality_outlier_report.csv",
        index=False,
    )

    rating_validity_report(eda_df).to_csv(
        reports_dir / "quality_rating_validity_report.csv",
        index=False,
    )

    duplicate_id_report(eda_df).to_csv(
        reports_dir / "quality_duplicate_id_report.csv",
        index=False,
    )

    title_quality_report(eda_df).to_csv(
        reports_dir / "quality_title_report.csv",
        index=False,
    )

    format_consistency_report(eda_df).to_csv(
        reports_dir / "quality_format_consistency_report.csv",
        index=False,
    )

    quality_df = full_quality_report(eda_df)

    save_quality_report(
        quality_df,
        output_path=str(reports_dir / "full_quality_report.csv"),
    )

    heatmap_path = charts_dir / "missing_values_heatmap.png"

    save_missing_heatmap(
        eda_df,
        output_path=str(heatmap_path),
    )

    if heatmap_path.exists():
        saved_charts.append(str(heatmap_path))

    logging.info("Lab 8 Part 7: Data quality reporting complete")

    upload_results = upload_lab8_charts_to_drive(saved_charts)

    if upload_results:
        pd.DataFrame(upload_results).to_csv(
            reports_dir / "google_drive_chart_uploads.csv",
            index=False,
        )

    logging.info("=== Lab 8 News Analytics Stage Complete ===")


def run_lab9_cleaning_stage():
    """
    Run Lab 9 cleaning on the raw news CSV generated by Lab 8.
    """
    logging.info("=== Lab 9 Cleaning Stage Started ===")

    raw_news_csv = Path("data/processed/analytics/raw_news_data.csv")

    if not raw_news_csv.exists():
        logging.warning(
            "Lab 9 raw input not found at %s. Running Lab 8 analytics first.",
            raw_news_csv,
        )
        run_lab8_analytics_stage()

    if not raw_news_csv.exists():
        raise FileNotFoundError(
            f"Lab 9 cleaning input does not exist: {raw_news_csv}"
        )

    cleaned_df = run_cleaning_pipeline_from_csv(
        input_path=str(raw_news_csv),
        save=True,
    )

    logging.info(
        "Lab 9 cleaned dataset shape: rows=%d columns=%d",
        cleaned_df.shape[0],
        cleaned_df.shape[1],
    )

    logging.info("=== Lab 9 Cleaning Stage Complete ===")

    return cleaned_df


def run_lab10_analysis_stage():
    """
    Run Lab 10 advanced data analysis on the Lab 9 cleaned dataset.

    This stage covers:
    - MySQL connection and table population
    - combining MySQL metrics with cleaned metadata
    - join type comparison
    - reshaping with melt()
    - pivot tables and crosstabs
    - groupby analysis
    - time series analysis
    - MongoDB aggregation pipelines
    - analytical insight reports and charts
    """
    logging.info("=== Lab 10 Advanced Analysis Stage Started ===")

    output_dir = Path("data/processed/analytics/lab10")
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_csv_path = Path("data/processed/cleaned/cleaned_data.csv")

    if not cleaned_csv_path.exists():
        logging.warning(
            "Lab 10 cleaned input not found at %s. Running Lab 9 cleaning first.",
            cleaned_csv_path,
        )
        run_lab9_cleaning_stage()

    if not cleaned_csv_path.exists():
        raise FileNotFoundError(
            f"Lab 10 input does not exist: {cleaned_csv_path}"
        )

    logging.info("Lab 10 Part 1: MySQL setup started")

    cleaned_df = pd.read_csv(cleaned_csv_path)

    mysql_metrics_df = setup_mysql_from_cleaned_data(
        cleaned_csv_path=str(cleaned_csv_path),
    )

    mysql_metrics_df.to_csv(
        output_dir / "mysql_news_article_metrics.csv",
        index=False,
    )

    logging.info(
        "Lab 10 MySQL metrics shape: rows=%d columns=%d",
        mysql_metrics_df.shape[0],
        mysql_metrics_df.shape[1],
    )

    logging.info("Lab 10 Part 1: MySQL setup complete")

    logging.info("Lab 10 Part 2: Combining data sources started")

    metadata_df = prepare_metadata_df(cleaned_df)

    combined_df = merge_metadata_with_metrics(
        metadata_df=metadata_df,
        metrics_df=mysql_metrics_df,
        how="left",
    )

    join_counts_df = compare_join_types(
        metadata_df=metadata_df,
        metrics_df=mysql_metrics_df,
    )

    save_join_count_chart(
        join_counts_df,
        output_path=str(output_dir / "join_type_row_counts.png"),
    )

    save_combined_outputs(
        combined_df,
        join_counts_df,
        output_dir=str(output_dir),
    )

    logging.info(
        "Lab 10 combined dataset shape: rows=%d columns=%d",
        combined_df.shape[0],
        combined_df.shape[1],
    )

    logging.info("Lab 10 Part 2: Combining data sources complete")

    logging.info("Lab 10 Part 3: Reshaping and pivot tables started")

    combined_df = add_primary_category(combined_df)
    combined_df = add_analysis_year(combined_df)

    long_metrics_df = wide_to_long_metrics(combined_df)
    wide_metrics_df = long_to_wide_metrics(long_metrics_df)

    category_year_pivot = build_category_year_pivot(combined_df)
    language_decade_crosstab = build_language_decade_crosstab(combined_df)

    wide_metrics_df.to_csv(
        output_dir / "news_metrics_wide_reconstructed.csv",
        index=False,
    )

    save_pivot_outputs(
        long_metrics_df,
        category_year_pivot,
        language_decade_crosstab,
        output_dir=str(output_dir),
    )

    logging.info(
        "Lab 10 reshaping complete: long=%s wide=%s pivot=%s",
        long_metrics_df.shape,
        wide_metrics_df.shape,
        category_year_pivot.shape,
    )

    logging.info("Lab 10 Part 3: Reshaping and pivot tables complete")

    logging.info("Lab 10 Part 4: GroupBy analysis started")

    category_df = category_summary(combined_df)
    document_df = document_type_summary(combined_df)
    yearly_df = yearly_trends(combined_df)
    top_articles_df = top_n_per_group(combined_df, n=3)
    transformed_df = add_group_average_columns(combined_df)
    filtered_large_categories_df = filter_large_categories(combined_df, min_count=5)

    transformed_df.to_csv(
        output_dir / "combined_with_group_averages.csv",
        index=False,
    )

    filtered_large_categories_df.to_csv(
        output_dir / "large_categories_only.csv",
        index=False,
    )

    save_yearly_trends_chart(
        yearly_df,
        output_path=str(output_dir / "yearly_trends.png"),
    )

    save_aggregation_outputs(
        category_df,
        yearly_df,
        top_articles_df,
        document_df,
        output_dir=str(output_dir),
    )

    logging.info("Lab 10 Part 4: GroupBy analysis complete")

    logging.info("Lab 10 Part 5: Time series analysis started")

    time_series_df = parse_news_dates(combined_df)
    time_series_df = add_date_components(time_series_df)

    monthly_df = build_monthly_time_series(time_series_df)
    yearly_time_series_df = resample_yearly(time_series_df)
    quarterly_df = resample_quarterly(time_series_df)

    monthly_rolling_df = add_rolling_averages(monthly_df)

    save_time_series_chart(
        monthly_rolling_df,
        output_path=str(output_dir / "rolling_estimated_value.png"),
    )

    save_time_series_outputs(
        monthly_rolling_df,
        yearly_time_series_df,
        quarterly_df,
        output_dir=str(output_dir),
    )

    logging.info("Lab 10 Part 5: Time series analysis complete")

    logging.info("Lab 10 Part 6: MongoDB aggregation started")

    mongo_category_df = run_category_aggregation()
    mongo_document_df = run_document_type_aggregation()
    mongo_source_df = run_source_aggregation()

    save_mongo_aggregation_outputs(
        mongo_category_df,
        mongo_document_df,
        mongo_source_df,
        output_dir=str(output_dir),
    )

    logging.info(
        "Lab 10 MongoDB aggregations complete: category=%s document=%s source=%s",
        mongo_category_df.shape,
        mongo_document_df.shape,
        mongo_source_df.shape,
    )

    logging.info("Lab 10 Part 6: MongoDB aggregation complete")

    logging.info("Lab 10 Part 7: Analytical insight reporting started")

    insight_df = generate_insight_questions(combined_df)

    insight_df.to_csv(
        output_dir / "analytical_questions_report.csv",
        index=False,
    )

    save_insight_outputs(
        combined_df,
        output_dir=str(output_dir),
    )

    logging.info(
        "Lab 10 generated %d analytical insights",
        len(insight_df),
    )

    logging.info("Lab 10 Part 7: Analytical insight reporting complete")

    logging.info("=== Lab 10 Advanced Analysis Stage Complete ===")

    return {
        "mysql_metrics_shape": mysql_metrics_df.shape,
        "combined_shape": combined_df.shape,
        "join_counts_shape": join_counts_df.shape,
        "long_metrics_shape": long_metrics_df.shape,
        "wide_metrics_shape": wide_metrics_df.shape,
        "pivot_shape": category_year_pivot.shape,
        "crosstab_shape": language_decade_crosstab.shape,
        "category_summary_shape": category_df.shape,
        "document_summary_shape": document_df.shape,
        "yearly_summary_shape": yearly_df.shape,
        "monthly_time_series_shape": monthly_rolling_df.shape,
        "yearly_time_series_shape": yearly_time_series_df.shape,
        "quarterly_time_series_shape": quarterly_df.shape,
        "mongo_category_shape": mongo_category_df.shape,
        "mongo_document_shape": mongo_document_df.shape,
        "mongo_source_shape": mongo_source_df.shape,
        "insights_shape": insight_df.shape,
    }


def run_pipeline():
    try:
        logging.info("Pipeline started")

        articles = fetch_news(query="technology", pages=3, page_size=5)
        logging.info(f"Fetched {len(articles)} total articles from API")

        parsed_articles = parse_json_files()
        logging.info(f"Parsed and stored {len(parsed_articles)} JSON articles to MongoDB")

        normal_pdf = "data/raw/pdf/news_normal.pdf"
        if Path(normal_pdf).exists():
            pdf_pages = extract_text_from_pdf(normal_pdf)
            for page in pdf_pages:
                save_to_mongo(
                    {"text": page["text"], "tables": page["tables"]},
                    page["source"],
                    {
                        "file_name": page["file_name"],
                        "document_type": page["document_type"],
                        "page_number": page["page_number"],
                        "extraction_timestamp": page["extraction_timestamp"],
                        "extraction_library": page["extraction_library"],
                    },
                )
            logging.info(f"Processed normal PDF: {normal_pdf}")

        two_column_pdf = "data/raw/pdf/news_two_column.pdf"
        if Path(two_column_pdf).exists():
            pdf_pages = extract_text_from_two_column_pdf(two_column_pdf)
            for page in pdf_pages:
                save_to_mongo(
                    {"text": page["text"], "tables": page["tables"]},
                    page["source"],
                    {
                        "file_name": page["file_name"],
                        "document_type": page["document_type"],
                        "page_number": page["page_number"],
                        "extraction_timestamp": page["extraction_timestamp"],
                        "extraction_library": page["extraction_library"],
                    },
                )
            logging.info(f"Processed two-column PDF: {two_column_pdf}")

        normal_word = "data/raw/word/news_normal.docx"
        if Path(normal_word).exists():
            word_data = extract_text_from_word(normal_word)
            save_to_mongo(
                {"text": word_data["text"], "tables": word_data["tables"]},
                word_data["source"],
                {
                    "file_name": word_data["file_name"],
                    "document_type": word_data["document_type"],
                    "extraction_timestamp": word_data["extraction_timestamp"],
                    "extraction_library": word_data["extraction_library"],
                },
            )
            logging.info(f"Processed normal Word file: {normal_word}")

        two_column_word = "data/raw/word/news_two_column.docx"
        if Path(two_column_word).exists():
            word_data = extract_text_from_two_column_word(two_column_word)
            save_to_mongo(
                {"text": word_data["text"], "tables": word_data["tables"]},
                word_data["source"],
                {
                    "file_name": word_data["file_name"],
                    "document_type": word_data["document_type"],
                    "extraction_timestamp": word_data["extraction_timestamp"],
                    "extraction_library": word_data["extraction_library"],
                },
            )
            logging.info(f"Processed two-column Word file: {two_column_word}")

        if Path(normal_word).exists():
            word_runs = extract_word_runs(normal_word)
            for run in word_runs:
                save_to_mongo(
                    run,
                    normal_word,
                    {
                        "file_name": Path(normal_word).name,
                        "document_type": "word_run",
                        "extraction_library": "python-docx",
                    },
                )
            logging.info(f"Processed Word runs for file: {normal_word}")

        excel_path = "data/raw/excel/news_data.xlsx"
        if Path(excel_path).exists():
            excel_articles = extract_data_from_excel(excel_path)
            for article in excel_articles:
                save_to_mongo(
                    article,
                    excel_path,
                    {
                        "file_name": "news_data.xlsx",
                        "document_type": "excel",
                        "extraction_library": "openpyxl",
                    },
                )

            excel_summary = extract_summary_from_excel(excel_path)
            save_to_mongo(
                excel_summary,
                excel_path,
                {
                    "file_name": "news_data.xlsx",
                    "document_type": "excel_summary",
                    "extraction_library": "openpyxl",
                },
            )
            logging.info(f"Processed Excel file: {excel_path}")

        encoding_file = "data/raw/api/news_page_1.json"
        if Path(encoding_file).exists():
            encoding_text = read_file_with_encoding(encoding_file)
            save_to_mongo(
                {"preview_text": encoding_text[:300]},
                encoding_file,
                {
                    "file_name": "news_page_1.json",
                    "document_type": "encoding_test",
                    "extraction_library": "chardet",
                },
            )
            logging.info(f"Encoding test processed for: {encoding_file}")

        hockey_url = "https://www.scrapethissite.com/pages/forms/"
        single_scraped = scrape_hockey_teams(hockey_url)
        for record in single_scraped:
            save_to_mongo(
                {
                    "name": record["name"],
                    "year": record["year"],
                    "wins": record["wins"],
                    "losses": record["losses"],
                },
                record["source"],
                {
                    "file_name": "hockey_results.json",
                    "document_type": "scraped_html",
                    "extraction_library": "requests_bs4",
                },
            )
        logging.info(f"Processed single-page scraping: {len(single_scraped)} records")

        multi_scraped = scrape_hockey_teams_multi_page(
            hockey_url,
            start_page=1,
            end_page=4,
        )
        for record in multi_scraped:
            save_to_mongo(
                {
                    "name": record["name"],
                    "year": record["year"],
                    "wins": record["wins"],
                    "losses": record["losses"],
                },
                record["source"],
                {
                    "file_name": "hockey_multi_page_results.json",
                    "document_type": "scraped_html_paginated",
                    "page_number": record.get("page"),
                    "extraction_library": "requests_bs4",
                },
            )
        logging.info(f"Processed multi-page scraping: {len(multi_scraped)} records")

        ajax_scraped = scrape_ajax_movies_api()
        for record in ajax_scraped:
            save_to_mongo(
                {
                    "title": record["title"],
                    "nominations": record["nominations"],
                    "awards": record["awards"],
                    "best_picture": record["best_picture"],
                    "year": record["year"],
                },
                record["source"],
                {
                    "file_name": "ajax_movies_api_results.json",
                    "document_type": "scraped_json_api",
                    "extraction_timestamp": record["extraction_timestamp"],
                    "extraction_library": "requests",
                },
            )
        logging.info(f"Processed dynamic JSON movie scraping: {len(ajax_scraped)} records")

        image_path = "data/raw/images/test_scan.png"
        if Path(image_path).exists():
            image_ocr = ocr_image(image_path)
            save_to_mongo(
                {
                    "raw_text": image_ocr["raw_text"],
                    "processed_text": image_ocr["processed_text"],
                },
                image_ocr["source"],
                {
                    "file_name": image_ocr["file_name"],
                    "document_type": image_ocr["type"],
                    "extraction_timestamp": image_ocr["extraction_timestamp"],
                    "extraction_library": "pytesseract",
                },
            )
            logging.info("Processed OCR image")

        scanned_pdf = "data/raw/scanned/test_scan.pdf"
        if Path(scanned_pdf).exists():
            pdf_ocr_results = ocr_scanned_pdf(scanned_pdf)
            for page in pdf_ocr_results:
                save_to_mongo(
                    {
                        "raw_text": page["raw_text"],
                        "processed_text": page["processed_text"],
                    },
                    page["source"],
                    {
                        "file_name": page["file_name"],
                        "document_type": page["type"],
                        "page_number": page["page_number"],
                        "extraction_timestamp": page["extraction_timestamp"],
                        "extraction_library": "pytesseract_pdf2image",
                    },
                )
            logging.info(f"Processed OCR scanned PDF: {len(pdf_ocr_results)} pages")

        image_articles = load_articles_from_json("data/raw/api")
        downloaded_images = download_article_images(
            image_articles,
            dest_dir="data/raw/images",
            limit=10,
        )
        logging.info(f"Downloaded {len(downloaded_images)} article images")

        image_results, image_errors = batch_process_images(
            input_dir="data/raw/images",
            output_dir="data/processed",
            max_width=500,
            thumb_size=(128, 128),
            convert_webp=True,
            extract_metadata=True,
            upload_to_drive=True,
        )
        logging.info(f"Batch processed {len(image_results)} images with {len(image_errors)} errors")

        save_batch_results_to_mongo(image_results)
        logging.info(f"Saved {len(image_results)} image metadata records to MongoDB")

        run_audio_video_stage()

        run_lab8_analytics_stage()

        run_lab9_cleaning_stage()

        run_lab10_analysis_stage()

        logging.info("Pipeline finished successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    run_pipeline()