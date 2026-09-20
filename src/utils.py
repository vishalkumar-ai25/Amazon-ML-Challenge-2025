import os
import re
import requests
from urllib.parse import urlparse
from pathlib import Path
from functools import partial
import multiprocessing
from tqdm import tqdm

PERMITTED_SCHEMES = {"http", "https"}


def is_permitted_url(url: object) -> bool:
    """Validate that the URL scheme is strictly http or https.
    
    Rejects file:, ftp:, custom schemes, or malformed inputs to prevent SSRF
    and local file inclusion vulnerabilities.
    """
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in PERMITTED_SCHEMES and bool(parsed.netloc)
    except Exception:
        return False


def download_image(image_link, savefolder, timeout=10):
    """Download an image using the secure requests client with scheme auditing."""
    if not is_permitted_url(image_link):
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(savefolder, filename)

    if os.path.exists(image_save_path) and os.path.getsize(image_save_path) > 0:
        return

    temp_path = f"{image_save_path}.tmp"
    try:
        with requests.get(image_link, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            os.replace(temp_path, image_save_path)
    except Exception as ex:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        print(f"Warning: Not able to download - {image_link}\n{ex}")


def download_images(image_links, download_folder):
    """Download images in parallel using multiprocessing and requests."""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    download_image_partial = partial(download_image, savefolder=download_folder)
    with multiprocessing.Pool(min(32, multiprocessing.cpu_count() or 4)) as pool:
        list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))