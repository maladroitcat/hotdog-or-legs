from pathlib import Path
import zipfile

from google.cloud import storage

from src.utils.config import load_config


def download_from_gcs(
    bucket_name: str,
    blob_path: str,
    local_path: str,
) -> Path:
    """
    Download a file from GCS to a local path if it doesn't already exist.
    Returns the local Path.
    """
    local_file = Path(local_path)
    local_file.parent.mkdir(parents=True, exist_ok=True)

    if local_file.exists():
        print(f"[data] Local file already exists at {local_file}, skipping download.")
        return local_file

    print(f"[data] Downloading gs://{bucket_name}/{blob_path} -> {local_file}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(str(local_file))
    print("[data] Download complete.")

    return local_file


def extract_zip(zip_path: str, extract_dir: str) -> Path:
    """
    Extract a zip file to the given directory.
    If the directory already has contents, skip extraction.
    """
    zip_file = Path(zip_path)
    out_dir = Path(extract_dir)

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[data] Extract directory {out_dir} is not empty, skipping extraction.")
        return out_dir

    print(f"[data] Extracting {zip_file} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(out_dir)

    print("[data] Extraction complete.")
    return out_dir


def main() -> None:
    """
    Entry point: read config.yaml, download dataset from GCS, extract it.
    """
    config = load_config()
    data_cfg = config["data"]

    bucket_name = data_cfg["bucket_name"]
    blob_path = data_cfg["blob_path"]
    local_zip_path = data_cfg["local_zip_path"]
    extract_dir = data_cfg["extract_dir"]

    zip_file = download_from_gcs(
        bucket_name=bucket_name,
        blob_path=blob_path,
        local_path=local_zip_path,
    )

    extract_zip(zip_path=str(zip_file), extract_dir=extract_dir)


if __name__ == "__main__":
    main()
