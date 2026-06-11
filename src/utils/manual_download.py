# src/utils/manual_download.py
import requests
import json
import os
from tqdm import tqdm

# The direct download URL for Touché 2020 dataset
# This is the official Zenodo link from the error message
url = "https://zenodo.org/record/3734893/files/debateorg.zip"

output_file = "data/raw/debateorg.zip"

print("=" * 60)
print("Downloading Touché 2020 dataset manually...")
print("=" * 60)

# Download with progress bar
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(output_file, 'wb') as f:
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

print(f"✅ Downloaded to {output_file}")
print("\nNow you need to extract the zip file and convert to JSON format.")