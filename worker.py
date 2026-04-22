import json
import re
import string
import io

import pika
import pandas as pd
import ray
from minio import Minio
from minio.error import S3Error

QUEUE_NAME = "preprocess_jobs"

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password"
MINIO_BUCKET = "processed-data"

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

try:
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
        print(f"Created MinIO bucket: {MINIO_BUCKET}")
except S3Error as e:
    print(f"MinIO setup error: {e}")
    raise


if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def augment_text(text):
    words = text.split()
    if len(words) > 4:
        words.pop(len(words) // 2)
    return " ".join(words)

@ray.remote
def process_chunk(chunk_df):
    chunk_df = chunk_df.copy()
    chunk_df["normalized_review"] = chunk_df["review"].apply(preprocess_text)
    chunk_df["augmented_review"] = chunk_df["normalized_review"].apply(augment_text)
    chunk_df["label"] = chunk_df["sentiment"].apply(
        lambda x: 1 if x == "positive" else 0
    )
    return chunk_df

def upload_to_minio(df, shard_id):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    data = csv_buffer.getvalue().encode("utf-8")

    object_name = f"processed_shard_{shard_id}.csv"

    minio_client.put_object(
        MINIO_BUCKET,
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type="text/csv"
    )

    return object_name

def split_dataframe(df, num_chunks):
    chunk_size = max(1, len(df) // num_chunks)
    chunks = []

    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i + chunk_size].copy())

    return chunks

def process_shard(job):
    input_file = job["input_file"]
    shard_id = job["shard_id"]
    start_row = job["start_row"]
    end_row = job["end_row"]

    df = pd.read_csv(input_file)
    shard_df = df.iloc[start_row:end_row].copy()

    if shard_df.empty:
        print(f"Shard {shard_id} is empty.")
        return

    chunks = split_dataframe(shard_df, num_chunks=4)

    futures = [process_chunk.remote(chunk) for chunk in chunks]
    processed_chunks = ray.get(futures)

    final_df = pd.concat(processed_chunks, ignore_index=True)
    final_df["shard_id"] = shard_id

    object_name = upload_to_minio(final_df, shard_id)

    print(
        f"Processed shard {shard_id}: rows {start_row}-{end_row}, "
        f"{len(final_df)} samples uploaded to MinIO as {object_name}"
    )

def callback(ch, method, properties, body):
    job = json.loads(body)
    print(f"Received job: {job}")

    try:
        process_shard(job)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"Error processing job {job.get('shard_id', 'unknown')}: {e}")

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host="localhost")
)
channel = connection.channel()

channel.queue_declare(queue=QUEUE_NAME, durable=True)
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

print("Waiting for preprocessing jobs...")
channel.start_consuming()