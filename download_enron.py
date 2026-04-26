import os
import shutil
import kagglehub
from pathlib import Path

# Fix the corruption error by explicitly wiping the cache directory first
cache_dir = Path.home() / ".cache/kagglehub/datasets/wcukierski/enron-email-dataset"
if cache_dir.exists():
    print("Clearing corrupted cache...")
    shutil.rmtree(cache_dir)

print("Downloading clean Enron dataset...")
# Download latest version securely
cached_path = kagglehub.dataset_download("wcukierski/enron-email-dataset")
print(f"Downloaded to cache: {cached_path}")

# Move it exactly where SONIC expects its datasets!
TARGET_DIR = Path(__file__).parent / "data" / "raw"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

target_file = TARGET_DIR / "emails.csv"
cached_file = Path(cached_path) / "emails.csv"

if cached_file.exists():
    print(f"Moving dataset into {TARGET_DIR} ...")
    shutil.copy2(cached_file, target_file)
    print(f"Success! Enron dataset is now perfectly situated at: {target_file}")
else:
    print("Could not find emails.csv in the downloaded cache.")
