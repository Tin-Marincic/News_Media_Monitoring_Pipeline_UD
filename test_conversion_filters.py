from pathlib import Path
from src.image_processing.processor import (
    convert_to_webp,
    convert_to_grayscale,
    save_optimised_jpeg,
    apply_blur,
    apply_sharpen,
    apply_edge_detection,
    enhance_contrast,
    enhance_brightness,
    enhance_color
)

image_files = list(Path("data/raw/images").glob("*"))

if not image_files:
    print("No images found in data/raw/images")
else:
    image_path = str(image_files[0])
    image_name = Path(image_path).stem

    print(f"Testing image: {image_path}\n")

    webp_output = f"data/processed/webp/{image_name}.webp"
    gray_output = f"data/processed/cropped/{image_name}_gray.jpg"
    jpeg_output = f"data/processed/resized/{image_name}_optimised.jpg"
    blur_output = f"data/processed/cropped/{image_name}_blur.jpg"
    sharpen_output = f"data/processed/cropped/{image_name}_sharpen.jpg"
    edges_output = f"data/processed/cropped/{image_name}_edges.jpg"
    contrast_output = f"data/processed/cropped/{image_name}_contrast.jpg"
    brightness_output = f"data/processed/cropped/{image_name}_bright.jpg"
    color_output = f"data/processed/cropped/{image_name}_color.jpg"

    convert_to_webp(image_path, webp_output)
    print(f"WebP saved to: {webp_output}")

    convert_to_grayscale(image_path, gray_output)
    print(f"Grayscale saved to: {gray_output}")

    save_optimised_jpeg(image_path, jpeg_output, quality=85)
    print(f"Optimised JPEG saved to: {jpeg_output}")

    apply_blur(image_path, blur_output, radius=3)
    print(f"Blur saved to: {blur_output}")

    apply_sharpen(image_path, sharpen_output)
    print(f"Sharpen saved to: {sharpen_output}")

    apply_edge_detection(image_path, edges_output)
    print(f"Edge detection saved to: {edges_output}")

    enhance_contrast(image_path, contrast_output, factor=1.5)
    print(f"Contrast enhancement saved to: {contrast_output}")

    enhance_brightness(image_path, brightness_output, factor=1.2)
    print(f"Brightness enhancement saved to: {brightness_output}")

    enhance_color(image_path, color_output, factor=1.3)
    print(f"Color enhancement saved to: {color_output}")