import logging
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

logger = logging.getLogger(__name__)


def extract_exif(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        exif_data = img.getexif()
        if not exif_data:
            logger.warning(f"No EXIF data found in {path}")
            return {}

        result = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, str(tag_id))

            # skip raw bytes completely
            if isinstance(value, bytes):
                continue

            # skip extremely long unreadable strings
            if isinstance(value, str) and len(value) > 200:
                continue

            result[tag_name] = value

        return result


def extract_gps(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        exif_data = img.getexif()
        if not exif_data:
            return None

        gps_ifd = exif_data.get_ifd(0x8825)
        if not gps_ifd:
            return None

        gps = {}
        for key, val in gps_ifd.items():
            gps_key = GPSTAGS.get(key, key)

            # skip unknown numeric GPS tags
            if isinstance(gps_key, int):
                continue

            # skip raw bytes
            if isinstance(val, bytes):
                continue

            # skip very long unreadable strings
            if isinstance(val, str) and len(val) > 200:
                continue

            gps[gps_key] = val

        return gps if gps else None


def get_exif_summary(path):
    exif = extract_exif(path)
    gps = extract_gps(path)

    return {
        "camera_make": exif.get("Make"),
        "camera_model": exif.get("Model"),
        "date_taken": exif.get("DateTimeOriginal") or exif.get("DateTime"),
        "exposure": str(exif.get("ExposureTime")) if exif.get("ExposureTime") is not None else None,
        "aperture": str(exif.get("FNumber")) if exif.get("FNumber") is not None else None,
        "iso": exif.get("ISOSpeedRatings"),
        "focal_length": str(exif.get("FocalLength")) if exif.get("FocalLength") is not None else None,
        "orientation": exif.get("Orientation"),
        "gps": gps,
    }


def strip_exif(path, output_path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        clean = Image.new(img.mode, img.size)
        clean.putdata(list(img.getdata()))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clean.save(output_path)

    logger.info(f"EXIF stripped: {output_path}")
    return str(output_path)