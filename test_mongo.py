from src.image_processing.batch import batch_process_images
from src.storage.mongo import save_batch_results_to_mongo, get_image_metadata

results, errors = batch_process_images(
    input_dir="data/raw/images",
    output_dir="data/processed",
    max_width=500,
    thumb_size=(128, 128),
    convert_webp=True,
    extract_metadata=True,
    upload_to_drive=True
)

save_batch_results_to_mongo(results)

print(f"Saved {len(results)} records to MongoDB")
print("Sample record:")
records = get_image_metadata()
if records:
    print(records[0])