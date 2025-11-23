from pathlib import Path
import shutil
import time
from typing import Dict, Iterable
from urllib.error import URLError

from bing_image_downloader import downloader
from PIL import Image, UnidentifiedImageError

DATA_ROOT = Path("local_data") / "hotdog_or_legs"
RAW_DIR = DATA_ROOT / "raw_bing"
PROCESSED_DIR = DATA_ROOT / "images"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLASS_QUERIES = {
   'hotdog': ['hotdogs poolside no bun', 'tan hotdogs', 'hotdogs that look like legs', 'hotdog or leg', 'hotdog or legs', 'hotdogs raw no bun', 'raw hotdog close up', 'two hotdogs'],
    'legs': ['human legs that look like hotdogs', 'bare human legs', 'human leg', 'legs in shorts', 'legs on beach', 'legs sunbathing'],
}

PER_QUERY_LIMIT = 60
MAX_EDGE = 400


def download_class_images(
    label: str,
    queries: Iterable[str],
    per_query_limit: int = PER_QUERY_LIMIT,
    max_retries: int = 3,
) -> None:
    """Download raw Bing image results for a single label."""
    base_dir = RAW_DIR / label
    shutil.rmtree(base_dir, ignore_errors=True)
    for query in queries:
        print(f'Downloading up to {per_query_limit} images for "{query}"')
        for attempt in range(max_retries):
            try:
                downloader.download(
                    query,
                    limit=per_query_limit,
                    output_dir=str(base_dir),
                    adult_filter_off=True,
                    force_replace=True,
                    timeout=30,
                    verbose=False,
                )
                break
            except URLError as exc:
                if attempt == max_retries - 1:
                    raise
                wait = 5 * (attempt + 1)
                print(f"  SSL/network issue ({exc}); retrying in {wait}s...")
                time.sleep(wait)


def clean_and_resize(label: str, max_edge: int = MAX_EDGE) -> int:
    """Convert every raw image to RGB JPEG and resize."""
    src_dir = RAW_DIR / label
    dest_dir = PROCESSED_DIR / label
    shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    for file in src_dir.rglob("*"):
        if not file.is_file():
            continue
        try:
            with Image.open(file) as img:
                img = img.convert("RGB")
                img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                out_path = dest_dir / f"{label}_{kept:04d}.jpg"
                img.save(out_path, format="JPEG", quality=90)
                kept += 1
        except (UnidentifiedImageError, OSError):
            dropped += 1
            continue
    print(f"{label}: kept {kept} images, dropped {dropped}")
    return kept


def build_dataset(class_queries: Dict[str, Iterable[str]] = CLASS_QUERIES) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label, queries in class_queries.items():
        download_class_images(label, queries)
        counts[label] = clean_and_resize(label)
    return counts


def summarize_counts(image_counts: Dict[str, int], sample_per_label: int = 5) -> None:
    for label, count in image_counts.items():
        sample_files = sorted((PROCESSED_DIR / label).glob("*.jpg"))[:sample_per_label]
        print(f"{label}: {count} images (showing {len(sample_files)} sample paths)")
        for path in sample_files:
            print("  ", path)


def main() -> None:
    counts = build_dataset()
    summarize_counts(counts)


if __name__ == "__main__":
    main()
