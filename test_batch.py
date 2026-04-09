from src.image_processing.batch import batch_process_images

results, errors = batch_process_images(
    input_dir="data/raw/images",
    output_dir="data/processed",
    max_width=500,
    thumb_size=(128, 128),
    convert_webp=True,
    extract_metadata=True,
    upload_to_drive=False
)

print(f"Processed {len(results)} images")
print(f"Errors: {len(errors)}")