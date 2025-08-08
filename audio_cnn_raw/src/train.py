import os
import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RawAudioFolderDataset, AudioPreprocConfig
from model import RawAudioCNN1D
from utils import seed_everything, compute_class_weights, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 1D CNN on raw audio for PASS/FAIL")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": acc}


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    seed_everything(args.seed)

    preproc = AudioPreprocConfig(
        target_sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        random_crop=True,
        normalize=True,
    )

    # Datasets
    train_ds = RawAudioFolderDataset(
        root_dir=args.train_dir,
        preproc=preproc,
        class_name_to_index={"fail": 0, "pass": 1},
        augment=True,
    )
    val_ds = RawAudioFolderDataset(
        root_dir=args.val_dir,
        preproc=AudioPreprocConfig(
            target_sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            random_crop=False,
            normalize=True,
        ),
        class_name_to_index=train_ds.get_label_mapping(),
        augment=False,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawAudioCNN1D(in_channels=1, num_classes=2, base_channels=args.base_channels, dropout=args.dropout)
    model.to(device)

    # Class weights for imbalance
    class_weights = compute_class_weights(train_ds.class_distribution()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")

    # Save label map for inference
    label_map_path = os.path.join(args.output_dir, "label_map.yaml")
    save_yaml(train_ds.get_label_mapping(), label_map_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for waveforms, labels in pbar:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                logits = model(waveforms)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(logits.detach(), dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item() * labels.size(0)

            pbar.set_postfix({
                "train_loss": f"{running_loss / max(running_total, 1):.4f}",
                "train_acc": f"{running_correct / max(running_total, 1):.4f}",
            })

        scheduler.step()

        # Validation
        metrics = evaluate(model, val_loader, device)
        val_acc = metrics["accuracy"]
        val_loss = metrics["loss"]
        print(f"\nValidation - loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Save best
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map_path": label_map_path,
                "args": vars(args),
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_ckpt_path)
            print(f"Saved best checkpoint to {best_ckpt_path}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()