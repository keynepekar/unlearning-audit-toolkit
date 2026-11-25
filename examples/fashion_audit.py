"""Audit a retain-only CNN update with a measurable Fashion-MNIST positive control."""

import argparse
import csv
import gzip
import json
import os
from copy import deepcopy
from hashlib import md5
from pathlib import Path
from struct import unpack_from
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from unlearning_audit.manifest import Manifest
from unlearning_audit.metrics import compare_losses
from unlearning_audit.validation import load_losses

BATCH = 256
FILES = {
    "train-images-idx3-ubyte.gz": ("8d4fb7e6c68d591d4c3dfef9ec88bf0d", (60000, 28, 28), 2051),
    "train-labels-idx1-ubyte.gz": ("25c81989df183df01b3e8a0aad5dffbe", (60000,), 2049),
    "t10k-images-idx3-ubyte.gz": ("bef4ecab320f06d8554ea6380940ec79", (10000, 28, 28), 2051),
    "t10k-labels-idx1-ubyte.gz": ("bb300cfdad3c16e7a12a480ee83cd310", (10000,), 2049),
}


def read_fashion(directory):
    arrays = []
    for name, (digest, shape, magic) in FILES.items():
        compressed = (directory / name).read_bytes()
        if md5(compressed, usedforsecurity=False).hexdigest() != digest:
            raise ValueError(f"Fashion-MNIST checksum mismatch: {name}")
        raw = gzip.decompress(compressed)
        header = (magic, *shape)
        if unpack_from(">" + "I" * len(header), raw) != header:
            raise ValueError(f"Invalid Fashion-MNIST header: {name}")
        arrays.append(np.frombuffer(raw, dtype=np.uint8, offset=4 * len(header)).reshape(shape))
    train_images, train_labels, test_images, test_labels = arrays
    return np.concatenate((train_images, test_images)), np.concatenate((train_labels, test_labels))


def split_rows(labels):
    rng = np.random.default_rng(20261125)
    by_class = [rng.choice(np.flatnonzero(labels[:60000] == label), 500, replace=False) for label in range(10)]
    forget = np.sort(np.concatenate([rows[:100] for rows in by_class]))
    retain = np.sort(np.concatenate([rows[100:] for rows in by_class]))
    return retain, forget, np.arange(60000, 70000)


def train(images, labels, rows, seed, epochs, model=None):
    torch.manual_seed(seed)
    if model is None:
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        ).to(images.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = torch.as_tensor(rng.permutation(rows), device=images.device)
        for batch in order.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(images[batch]), labels[batch]).backward()
            optimizer.step()
    return model


@torch.inference_mode()
def score(model, images, labels, rows):
    model.eval()
    losses, correct = [], []
    for batch in torch.as_tensor(rows, device=images.device).split(BATCH):
        logits = model(images[batch])
        expected = labels[batch]
        losses.append(F.cross_entropy(logits, expected, reduction="none"))
        correct.append(logits.argmax(1) == expected)
    return torch.cat(losses).cpu().tolist(), torch.cat(correct).float().mean().item()


def write_run(folder, models, images, labels, cohorts, seed):
    folder.mkdir()
    manifest = Manifest.from_dict({
        "schema_version": 1, "title": "Fashion-MNIST retain-only audit",
        "loss_definition": "Float32 categorical cross-entropy from logits",
        "models": {role: role for role in models},
        "cohorts": {"forget": "1,000 stratified official training rows", "holdout": "10,000 official test rows"},
        "provenance": {"Dataset": "Zalando Research Fashion-MNIST", "Protocol": "fashion-cnn-control-v1",
                       "Training seed": str(seed)},
    })
    accuracy = {}
    losses_path = folder / "losses.csv"
    with losses_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("sample_id", "cohort", "model", "loss"))
        for role, model in models.items():
            for cohort, rows in cohorts.items():
                losses, accuracy[f"{role}_{cohort}"] = score(model, images, labels, rows)
                writer.writerows((f"fashion-{row:05d}", cohort, role, loss)
                                 for row, loss in zip(rows, losses, strict=True))
    (folder / "manifest.json").write_text(json.dumps(manifest.as_dict(), indent=2) + "\n", encoding="utf-8")
    metrics = compare_losses(load_losses(losses_path), manifest, include_fingerprints=True)
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    auc = {role: model["auc"] for role, model in metrics["models"].items()}
    return {"seed": seed, "auc": auc, "deltas": metrics["auc_deltas"], "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seeds", type=int, default=6)
    if (args := parser.parse_args()).seeds < 1:
        parser.error("--seeds must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    pixels, labels = read_fashion(args.dataset)
    retain, forget, holdout = split_rows(labels)
    training = np.sort(np.concatenate((retain, forget)))
    images = torch.as_tensor(pixels[:, None]).to(device=args.device, dtype=torch.float32).div_(255)
    labels = torch.as_tensor(labels).to(device=args.device, dtype=torch.long)
    args.output.mkdir()
    runs = []
    started = perf_counter()
    for seed in range(42, 42 + args.seeds):
        before = train(images, labels, training, seed, 30)
        reference = train(images, labels, retain, seed, 30)
        candidate = train(images, labels, retain, seed + 20000, 10, deepcopy(before))
        run = write_run(args.output / f"seed-{seed}", {"before": before, "candidate": candidate,
                        "reference": reference}, images, labels, {"forget": forget, "holdout": holdout}, seed)
        runs.append(run)
        print(f"Seed {seed}: {run['auc']}", flush=True)
    summary = {"device": args.device, "torch": torch.__version__, "numpy": np.__version__,
               "seconds": perf_counter() - started, "runs": runs}
    if args.device == "cuda":
        summary.update(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
