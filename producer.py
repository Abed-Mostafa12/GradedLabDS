import json
import math

import pika
import pandas as pd

QUEUE_NAME = "preprocess_jobs"
INPUT_FILE = "IMDB Dataset.csv"
NUM_SHARDS = 4

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
        total_rows = len(df)
        print(f"Loaded {total_rows} rows from {INPUT_FILE}")
    except Exception as e:
        print(f"Could not read input file {INPUT_FILE}")
        print(e)
        return

    shard_size = math.ceil(total_rows / NUM_SHARDS)

    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host="localhost")
        )
        channel = connection.channel()
        print("Connected to RabbitMQ")
    except Exception as e:
        print("Could not connect to RabbitMQ on localhost:5672")
        print(e)
        return

    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.queue_purge(queue=QUEUE_NAME)

    jobs_sent = 0

    for shard_id in range(NUM_SHARDS):
        start_row = shard_id * shard_size
        end_row = min(start_row + shard_size, total_rows)

        if start_row >= total_rows:
            break

        job = {
            "shard_id": shard_id,
            "input_file": INPUT_FILE,
            "start_row": start_row,
            "end_row": end_row
        }

        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_NAME,
            body=json.dumps(job),
            properties=pika.BasicProperties(delivery_mode=2)
        )

        print(f"Sent job: {job}")
        jobs_sent += 1

    print(f"All {jobs_sent} shard jobs sent.")
    connection.close()

if __name__ == "__main__":
    main()