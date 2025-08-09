import os
import argparse
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from dotenv import load_dotenv

from dataset import RawAudioFolderDataset, AudioPreprocConfig
from model import RawAudioCNN1D
from model_adv import RawAudioResNet1D
from utils import seed_everything, compute_class_weights, save_yaml
from augment import mixup_waveforms, random_time_shift


# Load environment variables from .env if present
load_dotenv()
ENV_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
ENV_DURATION_SEC = float(os.getenv("AUDIO_DURATION_SEC", "4.0"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 1D CNN on raw audio for PASS/FAIL")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--sample_rate", type=int, default=ENV_SAMPLE_RATE)
    parser.add_argument("--duration_sec", type=float, default=ENV_DURATION_SEC)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_amp", action="store_true")
    # Advanced options
    parser.add_argument("--model_name", type=str, default="resnet", choices=["basic", "resnet", "resattn"], help="Model architecture")
    parser.add_argument("--time_shift_ms", type=int, default=50, help="Max time shift (+/- ms) for waveform roll during train")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup Beta alpha; 0 disables mixup")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing for CE; used when mixup disabled")
    parser.add_argument("--ema_decay", type=float, default=0.995, help="EMA decay; 0 disables EMA")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--clip_grad_norm", type=float, default=2.0, help="Gradient clipping max norm; 0 disables")
    parser.add_argument("--use_weighted_sampler", action="store_true", help="Use WeightedRandomSampler for class imbalance")
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


def build_model(name: str, num_classes: int, base_channels: int, dropout: float) -> nn.Module:
    if name == "basic":
        return RawAudioCNN1D(in_channels=1, num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    if name in ("resnet", "resattn"):
        return RawAudioResNet1D(num_classes=num_classes, base_channels=base_channels, dropout=dropout, use_attention=(name == "resattn"))
    raise ValueError(f"Unknown model_name: {name}")


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    if decay <= 0:
        return
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    if class_weights is not None:
        # weight per class
        weighted = soft_targets * log_probs * class_weights.unsqueeze(0)
    else:
        weighted = soft_targets * log_probs
    loss = -weighted.sum(dim=1).mean()
    return loss


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
    label_map = {"fail": 0, "pass": 1}
    train_ds = RawAudioFolderDataset(
        root_dir=args.train_dir,
        preproc=preproc,
        class_name_to_index=label_map,
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

    # Sampler for imbalance
    if args.use_weighted_sampler:
        counts = train_ds.class_distribution()
        total = sum(counts.values())
        class_weights = {cls: total / max(1, cnt) for cls, cnt in counts.items()}
        sample_weights = [class_weights[train_ds.labels[i]] for i in range(len(train_ds))]
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)
        shuffle_flag = False
    else:
        sampler = None
        shuffle_flag = True

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle_flag if sampler is None else False,
        sampler=sampler,
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
    model = build_model(args.model_name, num_classes=2, base_channels=args.base_channels, dropout=args.dropout)
    model.to(device)

    # Class weights for imbalance in hard-label CE
    class_weights_tensor = compute_class_weights(train_ds.class_distribution()).to(device)

    # Loss variants
    hard_ce = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # EMA model
    use_ema = args.ema_decay > 0
    ema_model = build_model(args.model_name, num_classes=2, base_channels=args.base_channels, dropout=args.dropout).to(device) if use_ema else None
    if use_ema:
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()

    best_val_acc = 0.0
    best_ema_val_acc = 0.0
    epochs_no_improve = 0

    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    best_ema_ckpt_path = os.path.join(args.output_dir, "best_model_ema.pt")

    # Save label map for inference
    label_map_path = os.path.join(args.output_dir, "label_map.yaml")
    save_yaml(train_ds.get_label_mapping(), label_map_path)

    # Time shift in samples
    max_shift_samples = int(args.time_shift_ms * args.sample_rate / 1000.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for waveforms, labels in pbar:
            waveforms = waveforms.to(device, non_blocking=True)  # [B, 1, T]
            labels = labels.to(device, non_blocking=True)        # [B]

            # Additional waveform-domain augmentations
            if max_shift_samples > 0:
                waveforms = random_time_shift(waveforms, max_shift_samples)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                # Mixup (produces soft labels when enabled)
                mixed_waveforms, soft_targets = mixup_waveforms(waveforms, labels, num_classes=2, alpha=args.mixup_alpha)
                logits = model(mixed_waveforms)
                if args.mixup_alpha > 0:
                    # Use soft cross-entropy; include class weights
                    loss = soft_cross_entropy(logits, soft_targets, class_weights_tensor)
                else:
                    loss = hard_ce(logits, labels)

            scaler.scale(loss).backward()
            if args.clip_grad_norm and args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if use_ema:
                update_ema(ema_model, model, args.ema_decay)

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                running_correct += (preds == labels).sum().item()
                running_total += labels.size(0)
                running_loss += loss.item() * labels.size(0)

            pbar.set_postfix({
                "train_loss": f"{running_loss / max(running_total, 1):.4f}",
                "train_acc": f"{running_correct / max(running_total, 1):.4f}",
            })

        scheduler.step()

        # Validation (standard model)
        metrics = evaluate(model, val_loader, device)
        val_acc = metrics["accuracy"]
        val_loss = metrics["loss"]
        print(f"\nValidation (model) - loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Validation (EMA model)
        if use_ema:
            ema_metrics = evaluate(ema_model, val_loader, device)
            ema_val_acc = ema_metrics["accuracy"]
            print(f"Validation (EMA)   - acc: {ema_val_acc:.4f}")
        else:
            ema_val_acc = 0.0

        improved = False
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map_path": label_map_path,
                "args": vars(args),
                "model_name": args.model_name,
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_ckpt_path)
            print(f"Saved best checkpoint to {best_ckpt_path}")
            improved = True

        if use_ema and ema_val_acc >= best_ema_val_acc:
            best_ema_val_acc = ema_val_acc
            torch.save({
                "model_state_dict": ema_model.state_dict(),
                "label_map_path": label_map_path,
                "args": vars(args),
                "model_name": args.model_name,
                "val_acc": ema_val_acc,
                "epoch": epoch,
                "ema": True,
            }, best_ema_ckpt_path)
            print(f"Saved best EMA checkpoint to {best_ema_ckpt_path}")
            improved = True

        if not improved:
            epochs_no_improve += 1
        else:
            epochs_no_improve = 0

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping after {epoch} epochs without improvement")
            break

    print(f"Best validation accuracy (model): {best_val_acc:.4f}")
    if use_ema:
        print(f"Best validation accuracy (EMA):   {best_ema_val_acc:.4f}")


if __name__ == "__main__":
    main()