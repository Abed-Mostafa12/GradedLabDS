import io
import random
import re
import string

import pandas as pd
import ray
from minio import Minio

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password"
RAW_BUCKET = "raw-data"
PROCESSED_BUCKET = "processed-data"

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

SYNONYMS = {
    "good": ["great", "nice", "excellent"],
    "bad": ["awful", "poor", "terrible"],
    "movie": ["film"],
    "show": ["series", "program"],
    "funny": ["humorous"],
    "boring": ["dull"]
}

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def augment_text(text: str, prob: float = 0.2) -> str:
    words = text.split()
    new_words = []

    for word in words:
        if word in SYNONYMS and random.random() < prob:
            new_words.append(random.choice(SYNONYMS[word]))
        else:
            new_words.append(word)

    return " ".join(new_words)

def upload_df_to_minio(df: pd.DataFrame, object_name: str) -> None:
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    data = csv_buffer.getvalue().encode("utf-8")

    minio_client.put_object(
        PROCESSED_BUCKET,
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type="text/csv"
    )

@ray.remote
def process_shard(shard_id: int, input_file: str, start_row: int, end_row: int) -> str:
    df = pd.read_csv(input_file)
    shard_df = df.iloc[start_row:end_row].copy()

    if shard_df.empty:
        return f"Shard {shard_id} empty"

    shard_df["normalized_review"] = shard_df["review"].apply(normalize_text)
    shard_df["augmented_review"] = shard_df["normalized_review"].apply(augment_text)
    shard_df["label"] = shard_df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    shard_df["shard_id"] = shard_id

    object_name = f"augmented_shard_{shard_id}.csv"
    upload_df_to_minio(shard_df, object_name)

    return f"Shard {shard_id} processed and uploaded as {object_name}"

def main():
    ray.init()

    input_file = "IMDB Dataset.csv"
    df = pd.read_csv(input_file)
    total_rows = len(df)
    num_shards = 4
    shard_size = (total_rows + num_shards - 1) // num_shards

    futures = []

    for shard_id in range(num_shards):
        start_row = shard_id * shard_size
        end_row = min(start_row + shard_size, total_rows)

        futures.append(
            process_shard.remote(shard_id, input_file, start_row, end_row)
        )

    results = ray.get(futures)

    for result in results:
        print(result)

    ray.shutdown()

if __name__ == "__main__":
    main()