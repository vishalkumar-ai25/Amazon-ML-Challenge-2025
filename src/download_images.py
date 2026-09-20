"""Concurrent, resilient product image downloader for Amazon ML Challenge 2025.

Downloads product images with:
- ThreadPoolExecutor for high-throughput I/O
- User-Agent header and socket timeouts to prevent stalled downloads
- Resumption support (skips already downloaded valid files)
- Progress tracking via tqdm
- Support for train and test splits
"""
from __future__ import annotations

import argparse
import os
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from tqdm import tqdm

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CONTEXT = ssl._create_unverified_context()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def download_single_image(
    image_url: str,
    save_path: str,
    *,
    timeout: int = 10,
) -> bool:
    """Download a single image file with timeout and resume check.

    Args:
        image_url: URL to download from.
        save_path: Destination file path.
        timeout: Socket timeout in seconds.

    Returns:
        True if successfully downloaded or already exists, False on error.
    """
    if not isinstance(image_url, str) or not image_url.startswith("http"):
        return False

    # Check if valid file already exists (> 1KB)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
        return True

    temp_path = f"{save_path}.tmp"
    try:
        req = urllib.request.Request(image_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as response:
            data = response.read()
            if len(data) > 0:
                with open(temp_path, "wb") as f:
                    f.write(data)
                os.replace(temp_path, save_path)
                return True
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return False

    return False


def download_dataset_images(
    df: pd.DataFrame,
    output_dir: str,
    *,
    max_workers: int = 32,
    limit: Optional[int] = None,
) -> dict[str, int]:
    """Download images for records in DataFrame concurrently.

    Saves images named by their sample_id: `{output_dir}/{sample_id}.jpg`
    to guarantee 1:1 mapping and prevent filename collisions.

    Args:
        df: DataFrame containing 'sample_id' and 'image_link'.
        output_dir: Folder to save downloaded images.
        max_workers: Concurrent download threads.
        limit: Optional limit on number of images to download.

    Returns:
        Dictionary with count of successful and failed downloads.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = df[["sample_id", "image_link"]].dropna().to_dict(orient="records")
    if limit is not None:
        records = records[:limit]

    success = 0
    failed = 0

    print(f"Downloading {len(records)} images to {output_dir} using {max_workers} threads...")

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for rec in records:
            sample_id = rec["sample_id"]
            url = rec["image_link"]
            save_path = os.path.join(output_dir, f"{sample_id}.jpg")
            tasks.append(executor.submit(download_single_image, url, save_path))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading"):
            if future.result():
                success += 1
            else:
                failed += 1

    print(f"Finished: {success} succeeded, {failed} failed.")
    return {"success": success, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Download product images")
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split to download (train, test, or both)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of download threads",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")

    if args.split in ["train", "both"]:
        train_csv = os.path.join(dataset_dir, "train.csv")
        if os.path.exists(train_csv):
            print("Loading train.csv...")
            train_df = pd.read_csv(train_csv, usecols=["sample_id", "image_link"])
            train_out = os.path.join(base_dir, "images", "train")
            download_dataset_images(train_df, train_out, max_workers=args.workers, limit=args.limit)

    if args.split in ["test", "both"]:
        test_csv = os.path.join(dataset_dir, "test.csv")
        if os.path.exists(test_csv):
            print("Loading test.csv...")
            test_df = pd.read_csv(test_csv, usecols=["sample_id", "image_link"])
            test_out = os.path.join(base_dir, "images", "test")
            download_dataset_images(test_df, test_out, max_workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
