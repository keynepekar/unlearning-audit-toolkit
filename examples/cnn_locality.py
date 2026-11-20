"""Compare a small CNN with retain-only refits on public UCI Digits data."""

import argparse
import csv
import json
import os
import platform
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from torch import nn
from torch.nn import functional as F

from digits_demo import split_examples
from sklearn_export import PROBABILITY_FLOOR
from unlearning_audit.manifest import Manifest
from unlearning_audit.metrics import compare_losses
from unlearning_audit.validation import load_losses

EPOCHS = 30
UPDATE_EPOCHS = 8
BATCH = 64
SEED = 42


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(128, 32), nn.ReLU(),
        )
        self.head = nn.Linear(32, 10)

    def forward(self, images):
        embedding = self.encoder(images)
        return self.head(embedding), embedding


def train(model, images, labels, rows, *, seed, epochs, learning_rate):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(rows), (len(rows) + BATCH - 1) // BATCH):
            batch = torch.as_tensor(indices, device=images.device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images[batch])
            F.cross_entropy(logits, labels[batch]).backward()
            optimizer.step()
    return model


def fit(images, labels, rows, seed):
    torch.manual_seed(seed)
    return train(
        DigitCNN().to(images.device), images, labels, rows,
        seed=seed, epochs=EPOCHS, learning_rate=0.003,
    )


@torch.no_grad()
def score(model, images, labels):
    logits, embeddings = model.eval()(images)
    probabilities = logits.softmax(dim=1)
    loss = -probabilities.gather(1, labels[:, None]).squeeze(1).clamp_min(PROBABILITY_FLOOR).log()
    return tuple(value.cpu().numpy().astype(float) for value in (probabilities, embeddings, loss))


def linear_cka(left, right):
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    denominator = np.linalg.norm(left.T @ left) * np.linalg.norm(right.T @ right)
    return float(np.linalg.norm(left.T @ right) ** 2 / denominator) if denominator else None


def distance_bands(pixels, holdout, forget):
    distances = np.linalg.norm(pixels[holdout, None, :] - pixels[forget], axis=2).min(axis=1)
    order = np.argsort(distances, kind="stable")
    return dict(zip(("near", "middle", "far"), np.array_split(order, 3), strict=True))


def compare_bands(scored, holdout, bands, seed):
    reference_prob, reference_repr, _ = scored["retrained"]
    for name in ("original", "candidate", "independent_refit"):
        probability, embedding, _ = scored[name]
        for band, positions in bands.items():
            rows = holdout[positions]
            gaps = np.abs(probability[rows] - reference_prob[rows]).sum(axis=1) / 2
            yield {
                "seed": seed, "model": name, "band": band, "rows": len(rows),
                "mean_total_variation": float(gaps.mean()),
                "linear_cka": linear_cka(embedding[rows], reference_repr[rows]),
            }


def write_audit(directory, scored, groups, seed):
    role_names = {"before": "original", "candidate": "candidate", "reference": "retrained"}
    cohorts = {kind: f"Fixed stratified UCI Digits {kind} partition." for kind in groups if kind != "retain"}
    losses = directory / "losses.csv"
    with losses.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("sample_id", "cohort", "model", "loss"))
        for name in role_names.values():
            for cohort in cohorts:
                writer.writerows((f"digit-{row:04d}", cohort, name, scored[name][2][row]) for row in groups[cohort])
    manifest = Manifest.from_dict({
        "schema_version": 1, "title": "CNN retain-only update on UCI Digits",
        "loss_definition": "Negative log of the true-class probability with a floor of 1e-15.",
        "models": role_names, "cohorts": cohorts,
        "provenance": {"Dataset": "UCI Optical Recognition of Handwritten Digits, DOI 10.24432/C50P49.",
                       "Training seed": str(seed), "Protocol": "cnn-digits-retain-only-v1"},
    })
    (directory / "manifest.json").write_text(json.dumps(manifest.as_dict(), indent=2) + "\n", encoding="utf-8")
    result = compare_losses(load_losses(losses), manifest, include_fingerprints=True)
    (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def run(directory, device, seeds):
    directory.mkdir()
    pixels, labels = load_digits(return_X_y=True)
    groups = split_examples(pixels, labels)
    bands = distance_bands(pixels / 16, groups["holdout"], groups["forget"])
    images = torch.as_tensor(pixels.reshape(-1, 1, 8, 8) / 16, dtype=torch.float32, device=device)
    targets = torch.as_tensor(labels, device=device, dtype=torch.long)
    all_train = np.concatenate((groups["retain"], groups["forget"]))
    rows = []
    accuracy = {}
    started = perf_counter()
    for seed in seeds:
        original = fit(images, targets, all_train, seed)
        candidate = train(
            deepcopy(original), images, targets, groups["retain"],
            seed=seed + 20_000, epochs=UPDATE_EPOCHS, learning_rate=0.001,
        )
        models = {
            "original": original, "candidate": candidate,
            "retrained": fit(images, targets, groups["retain"], seed),
            "independent_refit": fit(images, targets, groups["retain"], seed + 10_000),
        }
        scored = {name: score(model, images, targets) for name, model in models.items()}
        rows.extend(compare_bands(scored, groups["holdout"], bands, seed))
        holdout = groups["holdout"]
        accuracy[seed] = {name: float(np.mean(values[0][holdout].argmax(1) == labels[holdout]))
                          for name, values in scored.items()}
        if seed == seeds.start:
            write_audit(directory, scored, groups, seed)
        print(f"Completed training seed {seed}", flush=True)
    with (directory / "locality.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (directory / "run.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "device": str(device), "duration_seconds": perf_counter() - started,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "seeds": list(seeds), "cohort_counts": {key: len(value) for key, value in groups.items()},
        "holdout_accuracy": accuracy,
    }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", type=int, default=24)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    run(args.output, torch.device(args.device), range(SEED, SEED + args.seeds))


if __name__ == "__main__":
    main()
