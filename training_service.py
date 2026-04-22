from __future__ import annotations

import io
import json
import os
import secrets
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import FileResponse
from minio import Minio
from pydantic import BaseModel, Field, field_validator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

from cnnModel import TextCNN
from train_dataset import prepare_dataset

# --------------------------------------------------
# Config
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
MODELS_DIR = RUNS_DIR / "models"
METRICS_DIR = RUNS_DIR / "metrics"
JOBS_DIR = RUNS_DIR / "jobs"

for directory in (RUNS_DIR, MODELS_DIR, METRICS_DIR, JOBS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = "29611"
os.environ.setdefault("MASTER_ADDR", DEFAULT_MASTER_ADDR)
os.environ.setdefault("MASTER_PORT", DEFAULT_MASTER_PORT)
os.environ.setdefault("USE_LIBUV", "0")

# Comma-separated tokens can be supplied through env, e.g.
# TRAINING_API_TOKENS=clientA-token,clientB-token
RAW_TOKENS = os.getenv("TRAINING_API_TOKENS", "dev-token-123")
VALID_API_TOKENS = {token.strip() for token in RAW_TOKENS.split(",") if token.strip()}

# Sliding-window rate limiting: max N requests per token per minute.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
RATE_LIMIT_WINDOW_SECONDS = 60
REQUEST_LOG: dict[str, deque[float]] = defaultdict(deque)
REQUEST_LOG_LOCK = threading.Lock()
JOB_LOCK = threading.Lock()
ACTIVE_THREADS: dict[str, threading.Thread] = {}


# --------------------------------------------------
# API models
# --------------------------------------------------
class SubmitJobRequest(BaseModel):
    batch_size: int = Field(default=32, ge=1, le=1024)
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    epochs: int = Field(default=5, ge=1, le=100)
    world_size: int = Field(default=3, ge=1, le=8)
    max_len: int = Field(default=100, ge=8, le=2048)

    @field_validator("world_size")
    @classmethod
    def world_size_reasonable(cls, value: int) -> int:
        if value > os.cpu_count() and os.cpu_count() is not None:
            raise ValueError("world_size cannot exceed available CPU cores on this machine")
        return value


class JobSubmittedResponse(BaseModel):
    job_id: str
    status: str
    created_at: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    config: dict[str, Any]
    error: str | None = None
    model_path: str | None = None
    metrics_path: str | None = None


class RunSummary(BaseModel):
    job_id: str
    status: Literal["completed", "failed"]
    created_at: str
    finished_at: str | None = None
    model_path: str | None = None
    metrics_path: str | None = None


class MetricsResponse(BaseModel):
    job_id: str
    status: str
    metrics: list[dict[str, Any]]


# --------------------------------------------------
# Persistence helpers
# --------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_file(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def metrics_file(job_id: str) -> Path:
    return METRICS_DIR / f"{job_id}.jsonl"


def model_file(job_id: str) -> Path:
    return MODELS_DIR / f"{job_id}.pth"


def save_job_record(record: dict[str, Any]) -> None:
    with job_file(record["job_id"]).open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def load_job_record(job_id: str) -> dict[str, Any]:
    path = job_file(job_id)
    if not path.exists():
        raise FileNotFoundError(job_id)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_job_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(JOBS_DIR.glob("*.json"), reverse=True):
        with path.open("r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def append_metric(job_id: str, payload: dict[str, Any]) -> None:
    with metrics_file(job_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def load_metrics(job_id: str) -> list[dict[str, Any]]:
    path = metrics_file(job_id)
    if not path.exists():
        return []
    metrics: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


# --------------------------------------------------
# Auth + rate limit
# --------------------------------------------------
def require_token(authorization: str | None = Header(default=None)) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must use Bearer token",
        )

    if token not in VALID_API_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
        )

    return token


def enforce_rate_limit(token: str = Depends(require_token)) -> str:
    now = time.time()
    with REQUEST_LOG_LOCK:
        bucket = REQUEST_LOG[token]
        while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()

        if len(bucket) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: max {RATE_LIMIT_PER_MINUTE} requests per minute",
            )

        bucket.append(now)

    return token


# --------------------------------------------------
# Training implementation
# --------------------------------------------------
@dataclass
class TrainConfig:
    job_id: str
    batch_size: int
    learning_rate: float
    epochs: int
    world_size: int
    max_len: int
    model_path: str
    metrics_path: str


def setup_process(rank: int, world_size: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = DEFAULT_MASTER_ADDR
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )


def cleanup_process() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def train_worker(rank: int, config_dict: dict[str, Any]) -> None:
    config = TrainConfig(**config_dict)
    port = str(29600 + (abs(hash(config.job_id)) % 2000))
    setup_process(rank, config.world_size, port)

    device = torch.device("cpu")

    try:
        dataset, vocab = prepare_dataset(max_len=config.max_len)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        split_generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=split_generator
        )

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=config.world_size,
            rank=rank,
            shuffle=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=0,
        )

        model = TextCNN(vocab_size=len(vocab)).to(device)
        ddp_model = DDP(model)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=config.learning_rate)

        for epoch in range(config.epochs):
            train_sampler.set_epoch(epoch)
            ddp_model.train()

            running_loss = 0.0
            batch_count = 0
            train_samples_processed = 0
            epoch_start = time.perf_counter()

            for texts, labels in train_loader:
                texts = texts.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = ddp_model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                batch_count += 1
                train_samples_processed += int(labels.size(0))

            epoch_seconds = max(time.perf_counter() - epoch_start, 1e-8)
            train_loss = running_loss / max(batch_count, 1)
            throughput = train_samples_processed / epoch_seconds

            ddp_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for texts, labels in val_loader:
                    texts = texts.to(device)
                    labels = labels.to(device)
                    outputs = ddp_model(texts)
                    predictions = torch.argmax(outputs, dim=1)
                    correct += int((predictions == labels).sum().item())
                    total += int(labels.size(0))

            val_acc = (correct / total) if total > 0 else 0.0

            append_metric(
                config.job_id,
                {
                    "job_id": config.job_id,
                    "epoch": epoch + 1,
                    "rank": rank,
                    "train_loss": train_loss,
                    "val_accuracy": val_acc,
                    "throughput_samples_per_sec": throughput,
                    "timestamp": utc_now(),
                },
            )

            loss_tensor = torch.tensor(train_loss, dtype=torch.float32)
            correct_tensor = torch.tensor(correct, dtype=torch.float32)
            total_tensor = torch.tensor(total, dtype=torch.float32)
            throughput_tensor = torch.tensor(throughput, dtype=torch.float32)

            dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(total_tensor, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(throughput_tensor, dst=0, op=dist.ReduceOp.SUM)

            if rank == 0:
                append_metric(
                    config.job_id,
                    {
                        "job_id": config.job_id,
                        "epoch": epoch + 1,
                        "rank": "global",
                        "train_loss": (loss_tensor / config.world_size).item(),
                        "val_accuracy": (correct_tensor / total_tensor).item() if total_tensor.item() > 0 else 0.0,
                        "throughput_samples_per_sec": (throughput_tensor / config.world_size).item(),
                        "timestamp": utc_now(),
                    },
                )

        if rank == 0:
            torch.save(ddp_model.module.state_dict(), config.model_path)

    finally:
        cleanup_process()


# --------------------------------------------------
# Job orchestration
# --------------------------------------------------
def run_training_job(job_id: str) -> None:
    with JOB_LOCK:
        record = load_job_record(job_id)
        record["status"] = "running"
        record["started_at"] = utc_now()
        save_job_record(record)

    try:
        config = TrainConfig(
            job_id=job_id,
            batch_size=int(record["config"]["batch_size"]),
            learning_rate=float(record["config"]["learning_rate"]),
            epochs=int(record["config"]["epochs"]),
            world_size=int(record["config"]["world_size"]),
            max_len=int(record["config"]["max_len"]),
            model_path=str(model_file(job_id)),
            metrics_path=str(metrics_file(job_id)),
        )

        if metrics_file(job_id).exists():
            metrics_file(job_id).unlink()

        mp.set_start_method("spawn", force=True)
        mp.spawn(
            train_worker,
            args=(asdict(config),),
            nprocs=config.world_size,
            join=True,
        )

        with JOB_LOCK:
            record = load_job_record(job_id)
            record["status"] = "completed"
            record["finished_at"] = utc_now()
            record["model_path"] = str(model_file(job_id))
            record["metrics_path"] = str(metrics_file(job_id))
            save_job_record(record)

    except Exception as exc:
        with JOB_LOCK:
            record = load_job_record(job_id)
            record["status"] = "failed"
            record["finished_at"] = utc_now()
            record["error"] = repr(exc)
            save_job_record(record)
    finally:
        with JOB_LOCK:
            ACTIVE_THREADS.pop(job_id, None)


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="Distributed Training Service",
    version="1.1.0",
    description="Submit and monitor CNN DDP training jobs with token auth and rate limiting.",
)


@app.get("/")
def root(_: str = Depends(enforce_rate_limit)) -> dict[str, str]:
    return {"message": "Training service is running"}


@app.post(
    "/jobs",
    response_model=JobSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_training_job(payload: SubmitJobRequest, _: str = Depends(enforce_rate_limit)) -> JobSubmittedResponse:
    job_id = uuid.uuid4().hex
    record = {
        "job_id": job_id,
        "status": "queued",
        "created_at": utc_now(),
        "started_at": None,
        "finished_at": None,
        "config": payload.model_dump(),
        "error": None,
        "model_path": None,
        "metrics_path": None,
    }
    save_job_record(record)

    thread = threading.Thread(target=run_training_job, args=(job_id,), daemon=True)
    ACTIVE_THREADS[job_id] = thread
    thread.start()

    return JobSubmittedResponse(job_id=job_id, status="queued", created_at=record["created_at"])


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, _: str = Depends(enforce_rate_limit)) -> JobStatusResponse:
    try:
        record = load_job_record(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobStatusResponse(**record)


@app.get("/runs", response_model=list[RunSummary])
def list_completed_runs(_: str = Depends(enforce_rate_limit)) -> list[RunSummary]:
    records = [r for r in list_job_records() if r["status"] in {"completed", "failed"}]
    return [
        RunSummary(
            job_id=r["job_id"],
            status=r["status"],
            created_at=r["created_at"],
            finished_at=r.get("finished_at"),
            model_path=r.get("model_path"),
            metrics_path=r.get("metrics_path"),
        )
        for r in records
    ]


@app.get("/runs/{job_id}/metrics", response_model=MetricsResponse)
def fetch_metrics(job_id: str, _: str = Depends(enforce_rate_limit)) -> MetricsResponse:
    try:
        record = load_job_record(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    if record["status"] not in {"running", "completed", "failed"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Run has not started")

    return MetricsResponse(job_id=job_id, status=record["status"], metrics=load_metrics(job_id))


@app.get("/runs/{job_id}/model")
def download_model(job_id: str, _: str = Depends(enforce_rate_limit)) -> FileResponse:
    try:
        record = load_job_record(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    if record["status"] != "completed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Model is not available for this run")

    path = model_file(job_id)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model file missing")

    return FileResponse(
        path=path,
        media_type="application/octet-stream",
        filename=path.name,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
