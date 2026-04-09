from pathlib import Path
from src.image_processing.exif_utils import extract_exif, extract_gps, get_exif_summary, strip_exif

sample_files = list(Path("data/raw/exif_samples").glob("*"))

if not sample_files:
    print("No EXIF sample image found in data/raw/exif_samples")
else:
    for file_path in sample_files:
        image_path = str(file_path)
        image_name = file_path.stem

        print("=" * 60)
        print(f"Testing EXIF on: {image_path}\n")

        exif = extract_exif(image_path)
        gps = extract_gps(image_path)
        summary = get_exif_summary(image_path)
        clean_output = f"data/processed/cropped/{image_name}_no_exif.jpg"

        print("FULL EXIF (cleaned):")
        for key, value in exif.items():
            if value is not None:
                print(f"{key}: {value}")

        print("\nGPS:")
        if gps:
            for key, value in gps.items():
                print(f"{key}: {value}")
        else:
            print("No GPS data")

        print("\nEXIF SUMMARY:")
        for key, value in summary.items():
            if value is not None:
                print(f"{key}: {value}")

        stripped_path = strip_exif(image_path, clean_output)
        print(f"\nClean image without EXIF saved to: {stripped_path}\n")