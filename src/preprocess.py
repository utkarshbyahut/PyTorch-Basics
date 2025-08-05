import os
import zipfile
import requests
from tqdm import tqdm
import pandas as pd
from utils import extract_label_from_filename


DATASET_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
DATA_DIR = "data/ravdess"
ZIP_PATH = os.path.join(DATA_DIR, "ravdess.zip")

def generate_dataset_csv(audio_dir, output_csv):
    rows = []
    for root, _, files in os.walk(audio_dir):
        for f in files:
            if f.endswith(".wav"):
                path = os.path.join(root, f)
                label = extract_label_from_filename(f)
                if label:  # Only add if valid
                    rows.append({"path": path, "label": label})
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to {output_csv}")

"""
# This script downloads the RAVDESS dataset, extracts it, and generates a CSV file with audio paths and labels.

def download_ravdess():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH):
        print("RAVDESS already downloaded.")
        return
    print("Downloading RAVDESS...")
    with requests.get(DATASET_URL, stream=True) as r:
        r.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)

    print("Download complete. Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete.")

"""    
if __name__ == "__main__":
    # download_ravdess()

    audio_path = DATA_DIR
    output_csv = os.path.join(DATA_DIR, "ravdess.csv")
    generate_dataset_csv(audio_path, output_csv)

