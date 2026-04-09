import logging
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)


def inspect_image(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        width, height = img.size
        file_size = path.stat().st_size

        return {
            "filename": path.name,
            "format": img.format,
            "mode": img.mode,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 3),
            "file_size_bytes": file_size,
            "file_size_kb": round(file_size / 1024, 1),
        }


def resize_image(path, output_path, width, height, resample=Image.Resampling.LANCZOS):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        resized = img.resize((width, height), resample)

        if str(output_path).lower().endswith((".jpg", ".jpeg")) and resized.mode in ("RGBA", "LA", "P"):
            resized = resized.convert("RGB")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        resized.save(output_path)

    logger.info(f"Resized {path} -> {output_path} ({width}x{height})")
    return output_path


def resize_proportional(path, output_path, max_width=500):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        w, h = img.size
        ratio = max_width / w
        new_h = int(h * ratio)
        resized = img.resize((max_width, new_h), Image.Resampling.LANCZOS)

        if str(output_path).lower().endswith((".jpg", ".jpeg")) and resized.mode in ("RGBA", "LA", "P"):
            resized = resized.convert("RGB")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        resized.save(output_path)

    logger.info(f"Resized proportionally: {w}x{h} -> {max_width}x{new_h}")
    return output_path


def generate_thumbnail(path, output_path, max_size=(128, 128)):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        thumb = img.copy()
        thumb.thumbnail(max_size, Image.Resampling.LANCZOS)

        if str(output_path).lower().endswith((".jpg", ".jpeg")) and thumb.mode in ("RGBA", "LA", "P"):
            thumb = thumb.convert("RGB")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        thumb.save(output_path)

    logger.info(f"Thumbnail saved: {output_path} {thumb.size}")
    return output_path


def generate_fixed_thumbnail(path, output_path, size=(128, 128), method="pad", bg_color="black"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        img = img.convert("RGB")

        if method == "fit":
            result = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
        elif method == "pad":
            result = ImageOps.pad(img, size, color=bg_color)
        elif method == "contain":
            result = ImageOps.contain(img, size)
        else:
            result = ImageOps.cover(img, size)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

    logger.info(f"Fixed thumbnail saved: {output_path}")
    return output_path


def crop_image(path, output_path, box):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        cropped = img.crop(box)

        if str(output_path).lower().endswith((".jpg", ".jpeg")) and cropped.mode in ("RGBA", "LA", "P"):
            cropped = cropped.convert("RGB")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)

    logger.info(f"Cropped {path} with box {box} -> {output_path}")
    return output_path


def crop_top_banner(path, output_path, banner_height=200):
    with Image.open(path) as img:
        width, height = img.size
        box = (0, 0, width, min(banner_height, height))
    return crop_image(path, output_path, box)


def crop_center_square(path, output_path):
    with Image.open(path) as img:
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        upper = (h - side) // 2
        box = (left, upper, left + side, upper + side)
    return crop_image(path, output_path, box)


def convert_to_webp(path, output_path, quality=85):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        img = img.convert("RGB")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "WEBP", quality=quality, optimize=True)

    orig_size = path.stat().st_size
    new_size = Path(output_path).stat().st_size
    saving_pct = round((1 - new_size / orig_size) * 100, 1)
    logger.info(f"WebP: {orig_size//1024}KB -> {new_size//1024}KB ({saving_pct}% saving)")
    return output_path


def convert_to_grayscale(path, output_path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        gray = img.convert("L")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        gray.save(output_path)

    return output_path


def save_optimised_jpeg(path, output_path, quality=85):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        img = img.convert("RGB")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "JPEG", quality=quality, optimize=True)

    return output_path


def apply_blur(path, output_path, radius=2):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))

        if str(output_path).lower().endswith((".jpg", ".jpeg")) and blurred.mode in ("RGBA", "LA", "P"):
            blurred = blurred.convert("RGB")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        blurred.save(output_path)

    return output_path


def apply_sharpen(path, output_path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        sharpened = img.filter(ImageFilter.SHARPEN)

        if str(output_path).lower().endswith((".jpg", ".jpeg")) and sharpened.mode in ("RGBA", "LA", "P"):
            sharpened = sharpened.convert("RGB")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sharpened.save(output_path)

    return output_path


def apply_edge_detection(path, output_path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        edges.save(output_path)

    return output_path


def enhance_contrast(path, output_path, factor=1.5):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        enhanced = ImageEnhance.Contrast(img).enhance(factor)

        if str(output_path).lower().endswith((".jpg", ".jpeg")) and enhanced.mode in ("RGBA", "LA", "P"):
            enhanced = enhanced.convert("RGB")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        enhanced.save(output_path)

    return output_path


def enhance_brightness(path, output_path, factor=1.2):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        enhanced = ImageEnhance.Brightness(img).enhance(factor)

        if str(output_path).lower().endswith((".jpg", ".jpeg")) and enhanced.mode in ("RGBA", "LA", "P"):
            enhanced = enhanced.convert("RGB")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        enhanced.save(output_path)

    return output_path


def enhance_color(path, output_path, factor=1.3):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        enhanced = ImageEnhance.Color(img).enhance(factor)

        if str(output_path).lower().endswith((".jpg", ".jpeg")) and enhanced.mode in ("RGBA", "LA", "P"):
            enhanced = enhanced.convert("RGB")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        enhanced.save(output_path)

    return output_path