from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass(frozen=True)
class ClipSample:
    path: Path
    class_id: int
    class_name: str


class VideoClipDataset(Dataset):
    def __init__(
        self,
        samples: list[ClipSample],
        frames_per_clip: int = 16,
        image_size: int = 112,
    ) -> None:
        self.samples = samples
        self.frames_per_clip = frames_per_clip
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def _read_clip(self, clip_path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(clip_path))
        frames: list[torch.Tensor] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            t = torch.from_numpy(frame).float() / 255.0
            t = t.permute(2, 0, 1)         
            frames.append(t)
        cap.release()

        if not frames:
            return torch.zeros(3, self.frames_per_clip, self.image_size, self.image_size)

        if len(frames) == 1:
            frames = frames * self.frames_per_clip

        if len(frames) >= self.frames_per_clip:
            idx = torch.linspace(0, len(frames) - 1, steps=self.frames_per_clip).long()
            sampled = [frames[int(i)] for i in idx]
        else:
            sampled = frames[:]
            while len(sampled) < self.frames_per_clip:
                sampled.append(frames[-1])

        clip = torch.stack(sampled, dim=1)           
        return clip

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        clip = self._read_clip(sample.path)
        label = torch.tensor(sample.class_id, dtype=torch.long)
        return clip, label


class Tiny3DClassifier(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = z.flatten(1)
        return self.head(z)


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Treino supervisionado temporal para clips GynSurg_Action_3sec.",
    )
    parser.add_argument(
        "--action-root",
        default=str(
            project_root
            / "data"
            / "gynsurg"
            / "GynSurg_Action_3sec"
            / "GynSurg_Action_3sec"
        ),
        help="Raiz contendo GynSurg_action_dataset, bleeding e smoke.",
    )
    parser.add_argument(
        "--tasks",
        default="action,bleeding,smoke",
        help="Lista separada por virgula: action, bleeding, smoke.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--frames-per-clip", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=0,
        help="Se >0, limita qtd de clips por classe (para treino rapido em CPU).",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_root / "models" / "action_events" / "runs"),
        help="Diretorio de saida de checkpoints e metricas.",
    )
    parser.add_argument(
        "--use-balanced-sampler",
        action="store_true",
        help="Usa WeightedRandomSampler no treino para reduzir viés de classe.",
    )
    parser.add_argument(
        "--use-class-weighted-loss",
        action="store_true",
        help="Aplica pesos inversos de classe na CrossEntropyLoss.",
    )
    parser.add_argument(
        "--checkpoint-every-epoch",
        action="store_true",
        help="Salva checkpoint por epoca (epoch_XXX.pt) em cada tarefa.",
    )
    return parser.parse_args()


def _dataset_dir_for_task(root: Path, task: str) -> Path:
    mapping = {
        "action": root / "GynSurg_action_dataset",
        "bleeding": root / "GynSurg_bleeding_dataset",
        "smoke": root / "GynSurg_smoke_dataset",
    }
    if task not in mapping:
        raise ValueError(f"Tarefa invalida: {task}")
    return mapping[task]


def _collect_samples(
    dataset_dir: Path,
    max_samples_per_class: int,
    seed: int,
) -> tuple[list[ClipSample], list[str]]:
    rng = random.Random(seed)
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    samples: list[ClipSample] = []
    for class_id, class_dir in enumerate(class_dirs):
        clips = sorted(class_dir.glob("*.mp4"))
        if max_samples_per_class > 0 and len(clips) > max_samples_per_class:
            rng.shuffle(clips)
            clips = clips[:max_samples_per_class]
        for clip_path in clips:
            samples.append(
                ClipSample(
                    path=clip_path,
                    class_id=class_id,
                    class_name=class_dir.name,
                )
            )
    return samples, class_names


def _stratified_split(
    samples: list[ClipSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[ClipSample], list[ClipSample]]:
    rng = random.Random(seed)
    buckets: dict[int, list[ClipSample]] = {}
    for s in samples:
        buckets.setdefault(s.class_id, []).append(s)

    train: list[ClipSample] = []
    val: list[ClipSample] = []
    for class_id, class_samples in buckets.items():
        rng.shuffle(class_samples)
        n_val = max(1, int(round(len(class_samples) * val_ratio)))
        n_val = min(n_val, max(1, len(class_samples) - 1))
        val.extend(class_samples[:n_val])
        train.extend(class_samples[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            logits = model(clips)
            loss = criterion(logits, labels)
            running_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
    if total == 0:
        return 0.0, 0.0
    return running_loss / total, correct / total


def _class_counts(samples: list[ClipSample], n_classes: int) -> list[int]:
    counts = [0 for _ in range(n_classes)]
    for sample in samples:
        counts[sample.class_id] += 1
    return counts


def _build_weighted_sampler(samples: list[ClipSample], n_classes: int) -> WeightedRandomSampler:
    counts = _class_counts(samples, n_classes)
    class_weights = [1.0 / max(c, 1) for c in counts]
    sample_weights = [class_weights[s.class_id] for s in samples]
    weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def _train_single_task(
    task: str,
    dataset_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    frames_per_clip: int,
    image_size: int,
    val_ratio: float,
    seed: int,
    max_samples_per_class: int,
    device: torch.device,
    use_balanced_sampler: bool,
    use_class_weighted_loss: bool,
    checkpoint_every_epoch: bool,
) -> None:
    samples, class_names = _collect_samples(dataset_dir, max_samples_per_class, seed)
    train_samples, val_samples = _stratified_split(samples, val_ratio, seed)

    train_ds = VideoClipDataset(train_samples, frames_per_clip=frames_per_clip, image_size=image_size)
    val_ds = VideoClipDataset(val_samples, frames_per_clip=frames_per_clip, image_size=image_size)
    if use_balanced_sampler:
        sampler = _build_weighted_sampler(train_samples, len(class_names))
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = Tiny3DClassifier(n_classes=len(class_names)).to(device)
    if use_class_weighted_loss:
        counts = _class_counts(train_samples, len(class_names))
        weights = torch.tensor(
            [len(train_samples) / max(len(class_names) * c, 1) for c in counts],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    history: list[dict[str, float]] = []

    print(f"\n[events:{task}] classes={class_names}")
    print(f"[events:{task}] train={len(train_samples)} val={len(val_samples)}")
    print(f"[events:{task}] balanced_sampler={use_balanced_sampler} weighted_loss={use_class_weighted_loss}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for clips, labels in train_loader:
            clips = clips.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = _evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )
        print(
            f"[events:{task}] epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if checkpoint_every_epoch:
            torch.save(
                {
                    "task": task,
                    "epoch": epoch,
                    "class_names": class_names,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "frames_per_clip": frames_per_clip,
                    "image_size": image_size,
                },
                output_dir / f"epoch_{epoch:03d}.pt",
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "task": task,
                    "class_names": class_names,
                    "model_state_dict": model.state_dict(),
                    "frames_per_clip": frames_per_clip,
                    "image_size": image_size,
                },
                output_dir / "best.pt",
            )
        scheduler.step()

    metrics = {
        "task": task,
        "classes": class_names,
        "epochs": epochs,
        "best_val_acc": best_val_acc,
        "history": history,
        "frames_per_clip": frames_per_clip,
        "image_size": image_size,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "task": task,
            "class_names": class_names,
            "model_state_dict": model.state_dict(),
            "frames_per_clip": frames_per_clip,
            "image_size": image_size,
        },
        output_dir / "last.pt",
    )
    print(f"[events:{task}] best_val_acc={best_val_acc:.4f} saved={output_dir / 'best.pt'}")


def main() -> int:
    args = _parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    action_root = Path(args.action_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[events] device={device}")

    for task in tasks:
        dataset_dir = _dataset_dir_for_task(action_root, task)
        if not dataset_dir.is_dir():
            raise RuntimeError(f"Dataset da tarefa nao encontrado: {dataset_dir}")
        _train_single_task(
            task=task,
            dataset_dir=dataset_dir,
            output_dir=output_root / task,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            frames_per_clip=args.frames_per_clip,
            image_size=args.image_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_samples_per_class=args.max_samples_per_class,
            device=device,
            use_balanced_sampler=bool(args.use_balanced_sampler),
            use_class_weighted_loss=bool(args.use_class_weighted_loss),
            checkpoint_every_epoch=bool(args.checkpoint_every_epoch),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
