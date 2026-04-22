import io
import pandas as pd
import torch
from torch.utils.data import Dataset
from minio import Minio

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password"
PROCESSED_BUCKET = "processed-data"

class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def load_shards_from_minio():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    objects = client.list_objects(PROCESSED_BUCKET, recursive=True)

    dfs = []
    for obj in objects:
        if obj.object_name.endswith(".csv"):
            response = client.get_object(PROCESSED_BUCKET, obj.object_name)
            data = response.read()
            df = pd.read_csv(io.BytesIO(data))
            dfs.append(df)

    if not dfs:
        raise ValueError("No processed shard CSVs found in MinIO")

    return pd.concat(dfs, ignore_index=True)

def tokenize(text):
    return str(text).split()

def build_vocab(texts, min_freq=1):
    word_freq = {}
    for text in texts:
        for token in tokenize(text):
            word_freq[token] = word_freq.get(token, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab

def encode_text(text, vocab, max_len=100):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_len]]

    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))

    return ids

def prepare_dataset(max_len=100):
    df = load_shards_from_minio()

    text_col = "augmented_review" if "augmented_review" in df.columns else "review"
    texts = df[text_col].tolist()
    labels = df["label"].tolist()

    vocab = build_vocab(texts)
    encoded_texts = [encode_text(text, vocab, max_len=max_len) for text in texts]

    texts_tensor = torch.tensor(encoded_texts, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = ReviewDataset(texts_tensor, labels_tensor)
    return dataset, vocab