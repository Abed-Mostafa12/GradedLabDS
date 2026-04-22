import os
import re
import string
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

class ReviewDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["review"]
        text = preprocess_text(text)
        label = 1 if self.df.iloc[idx]["sentiment"] == "positive" else 0
        return text, label

def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )

    dataset = ReviewDataset("IMDB Dataset.csv")

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=0
    )

    for epoch in range(2):
        sampler.set_epoch(epoch)
        print(f"\nRank {rank} | Epoch {epoch}")

        for batch_idx, (texts, labels) in enumerate(loader):
            print(f"Rank {rank} | Batch {batch_idx} | labels={labels.tolist()}")
            print(f"Rank {rank} | cleaned sample: {texts[0][:80]}")
            if batch_idx == 1:
                break

    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()