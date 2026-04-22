import os
import io
import time
import argparse
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from minio import Minio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from cnnModel import TextCNN


# ----------------------------
# Environment setup for Windows / local DDP
# ----------------------------
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29501"
os.environ["USE_LIBUV"] = "0"


# ----------------------------
# MinIO config
# ----------------------------
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password"
PROCESSED_BUCKET = "processed-data"


# ----------------------------
# Dataset helpers
# ----------------------------
def load_shards_from_minio():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    dataframes = []
    objects = client.list_objects(PROCESSED_BUCKET, recursive=True)

    for obj in objects:
        if obj.object_name.endswith(".csv"):
            response = client.get_object(PROCESSED_BUCKET, obj.object_name)
            try:
                data = response.read()
                df = pd.read_csv(io.BytesIO(data))
                dataframes.append(df)
            finally:
                response.close()
                response.release_conn()

    if not dataframes:
        raise ValueError("No processed shard CSVs found in MinIO bucket 'processed-data'.")

    return pd.concat(dataframes, ignore_index=True)


def tokenize(text):
    return str(text).split()


def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len=100):
    tokens = tokenize(text)
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_len]]

    if len(token_ids) < max_len:
        token_ids += [vocab["<PAD>"]] * (max_len - len(token_ids))

    return token_ids


class ReviewDataset(Dataset):
    def __init__(self, texts_tensor, labels_tensor):
        self.texts = texts_tensor
        self.labels = labels_tensor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def prepare_dataset(max_len=100):
    df = load_shards_from_minio()

    text_column = "augmented_review" if "augmented_review" in df.columns else "review"
    texts = df[text_column].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    vocab = build_vocab(texts)
    encoded_texts = [encode_text(text, vocab, max_len=max_len) for text in texts]

    texts_tensor = torch.tensor(encoded_texts, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = ReviewDataset(texts_tensor, labels_tensor)
    return dataset, vocab


# ----------------------------
# DDP setup
# ----------------------------
def setup_process(rank, world_size):
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )


def cleanup_process():
    if dist.is_initialized():
        dist.destroy_process_group()


# ----------------------------
# Argument parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Distributed CNN training with PyTorch DDP")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per worker")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--world_size", type=int, default=3, help="Number of DDP workers")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--model_path", type=str, default="textcnn_ddp.pth", help="Path to save trained model")

    return parser.parse_args()


# ----------------------------
# Training worker
# ----------------------------
def train_worker(rank, world_size, args):
    print(f"[Rank {rank}] Starting worker")
    setup_process(rank, world_size)
    print(f"[Rank {rank}] Process group initialized")

    device = torch.device("cpu")

    dataset, vocab = prepare_dataset(max_len=args.max_len)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    split_generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=0
    )

    model = TextCNN(vocab_size=len(vocab)).to(device)
    model = DDP(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        running_loss = 0.0
        batch_count = 0
        train_samples_processed = 0

        epoch_start_time = time.time()

        for texts, labels in train_loader:
            texts = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            train_samples_processed += labels.size(0)

        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = running_loss / max(batch_count, 1)
        throughput = train_samples_processed / max(epoch_duration, 1e-8)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for texts, labels in val_loader:
                texts = texts.to(device)
                labels = labels.to(device)

                outputs = model(texts)
                predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        local_val_acc = correct / total if total > 0 else 0.0

        # Per-worker logging
        print(
            f"[Rank {rank}] Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {local_val_acc:.4f} | "
            f"Throughput: {throughput:.2f} samples/sec"
        )

        # Global aggregated metrics on rank 0
        local_loss_tensor = torch.tensor(avg_train_loss, dtype=torch.float32)
        local_correct_tensor = torch.tensor(correct, dtype=torch.float32)
        local_total_tensor = torch.tensor(total, dtype=torch.float32)
        local_throughput_tensor = torch.tensor(throughput, dtype=torch.float32)

        dist.reduce(local_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(local_correct_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(local_total_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(local_throughput_tensor, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            global_loss = (local_loss_tensor / world_size).item()
            global_acc = (
                (local_correct_tensor / local_total_tensor).item()
                if local_total_tensor.item() > 0 else 0.0
            )
            avg_throughput = (local_throughput_tensor / world_size).item()

            print(
                f"[Global] Epoch {epoch + 1}/{args.epochs} | "
                f"Avg Train Loss: {global_loss:.4f} | "
                f"Global Val Acc: {global_acc:.4f} | "
                f"Avg Throughput: {avg_throughput:.2f} samples/sec"
            )

    if rank == 0:
        torch.save(model.module.state_dict(), args.model_path)
        print(f"[Rank 0] Model saved to {args.model_path}")

    cleanup_process()
    print(f"[Rank {rank}] Finished")


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    print("Starting distributed training job with:")
    print(f"  batch_size = {args.batch_size}")
    print(f"  lr         = {args.lr}")
    print(f"  epochs     = {args.epochs}")
    print(f"  world_size = {args.world_size}")
    print(f"  max_len    = {args.max_len}")
    print(f"  model_path = {args.model_path}")

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        train_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()