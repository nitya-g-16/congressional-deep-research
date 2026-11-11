import os
from pathlib import Path

import uvicorn
from google.cloud import storage

DATASET = Path(__file__).resolve().parent / "data_cleaned" / "house_individual_donations_pre_request.csv"

def prepare_data():
    if DATASET.exists():
        return

    print("Downloading congressional donation dataset from Cloud Storage...", flush=True)

    DATASET.parent.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket("ieor-4570-f26-ng3098-p2-data")
    blob = bucket.blob("house_individual_donations_pre_request.csv")
    blob.download_to_filename(str(DATASET))

    print("Dataset ready.", flush=True)

if __name__ == "__main__":
    prepare_data()
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
    )
