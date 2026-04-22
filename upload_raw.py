from minio import Minio
import os

minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="password",
    secure=False
)

BUCKET = "raw-data"
FILE_NAME = "IMDB Dataset.csv"

if not minio_client.bucket_exists(BUCKET):
    minio_client.make_bucket(BUCKET)

file_size = os.path.getsize(FILE_NAME)

with open(FILE_NAME, "rb") as file_data:
    minio_client.put_object(
        BUCKET,
        FILE_NAME,
        file_data,
        length=file_size,
        content_type="text/csv"
    )

print("Raw dataset uploaded to MinIO")