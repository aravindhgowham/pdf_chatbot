#!/usr/bin/env python3
import argparse
import os
import sys
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from scipy.signal import resample_poly
try:
    import soundfile as sf
    _HAVE_SF = True
except Exception:
    _HAVE_SF = False
    from scipy.io import wavfile

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False


# =========================
# Audio I/O and processing
# =========================

def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    if _HAVE_SF:
        audio, sr = sf.read(str(path), always_2d=False)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        return audio, sr
    else:
        sr, audio = wavfile.read(str(path))
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        else:
            audio = audio.astype(np.float32)
        return audio, sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1).astype(np.float32)


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    g = np.gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    return resample_poly(audio, up=up, down=down).astype(np.float32)


def normalize_audio(audio: np.ndarray, method: str = "rms", eps: float = 1e-8, remove_dc: bool = True) -> np.ndarray:
    if remove_dc:
        audio = audio - np.mean(audio)
    if method == "none":
        return audio.astype(np.float32)
    if method == "rms":
        rms = np.sqrt(np.mean(np.square(audio)) + eps)
        if rms > 0:
            audio = audio / rms
    elif method == "peak":
        peak = np.max(np.abs(audio)) + eps
        if peak > 0:
            audio = audio / peak
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return audio.astype(np.float32)


def chunk_audio(audio: np.ndarray, sr: int, window_seconds: float, hop_seconds: float, pad: str = "center") -> List[np.ndarray]:
    win = int(window_seconds * sr)
    hop = int(hop_seconds * sr)
    if win <= 0:
        return []
    chunks: List[np.ndarray] = []
    if len(audio) < win:
        pad_len = win - len(audio)
        if pad == "center":
            left = pad_len // 2
            right = pad_len - left
            padded = np.pad(audio, (left, right))
        elif pad == "right":
            padded = np.pad(audio, (0, pad_len))
        elif pad == "repeat":
            # Tile until at least win length, then crop
            reps = int(np.ceil(win / max(1, len(audio))))
            padded = np.tile(audio, reps)[:win]
        else:
            raise ValueError(f"Unknown pad: {pad}")
        chunks.append(padded.astype(np.float32))
        return chunks
    if hop <= 0:
        hop = win
    for start in range(0, len(audio) - win + 1, hop):
        end = start + win
        chunks.append(audio[start:end].astype(np.float32))
    return chunks


# =========================
# Model and inference
# =========================

def load_torchscript(model_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def infer_windows_batched(model, device: torch.device, windows: List[np.ndarray], batch_size: int) -> List[float]:
    probs: List[float] = []
    total = len(windows)
    rng = range(0, total, batch_size)
    if _HAVE_TQDM:
        rng = tqdm(rng, total=(total + batch_size - 1) // batch_size, desc="Batches")
    with torch.no_grad():
        for i in rng:
            batch = windows[i:i + batch_size]
            x = np.stack(batch, axis=0)  # [B, T]
            xt = torch.from_numpy(x).to(device).float()[:, None, :]  # [B, 1, T]
            y = model(xt)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            if y.shape[-1] == 1:
                p = torch.sigmoid(y).squeeze(-1)
            else:
                p = torch.softmax(y, dim=-1)[..., -1]
            probs.extend(p.detach().cpu().numpy().astype(np.float32).tolist())
    return probs


def aggregate_probs(probs: List[float], mode: str) -> float:
    if not probs:
        return 0.0
    if mode == "max":
        return float(np.max(probs))
    if mode == "mean":
        return float(np.mean(probs))
    if mode == "median":
        return float(np.median(probs))
    raise ValueError(f"Unknown aggregate: {mode}")


def collect_files(wav_dir: Path, files: List[Path], exts: List[str]) -> List[Path]:
    result: List[Path] = []
    if wav_dir is not None:
        for p in sorted(wav_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                result.append(p)
    for f in files or []:
        p = Path(f)
        if p.is_dir():
            for q in sorted(p.rglob("*")):
                if q.is_file() and q.suffix.lower() in exts:
                    result.append(q)
        elif p.is_file() and p.suffix.lower() in exts:
            result.append(p)
    # Ensure unique
    seen = set()
    unique: List[Path] = []
    for p in result:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def infer_many(
    model,
    device: torch.device,
    paths: List[Path],
    target_sr: int,
    window_s: float,
    hop_s: float,
    pad: str,
    norm: str,
    remove_dc: bool,
    batch_size: int,
    aggregate: str,
    invert: bool,
) -> List[Dict[str, Any]]:

    # Build all windows across files
    file_windows: List[List[int]] = []  # mapping from file idx -> list of indices into global_windows
    global_windows: List[np.ndarray] = []

    file_infos: List[Tuple[Path, int]] = []  # (path, num_windows)

    file_iter = paths
    if _HAVE_TQDM:
        file_iter = tqdm(paths, desc="Reading")
    for path in file_iter:
        audio, sr = read_audio(path)
        audio = to_mono(audio)
        audio = resample(audio, sr, target_sr)
        audio = normalize_audio(audio, method=norm, remove_dc=remove_dc)
        chunks = chunk_audio(audio, target_sr, window_s, hop_s, pad=pad)
        start_idx = len(global_windows)
        for c in chunks:
            global_windows.append(c)
        end_idx = len(global_windows)
        file_windows.append(list(range(start_idx, end_idx)))
        file_infos.append((path, len(chunks)))

    # Inference over all windows batched
    all_probs = infer_windows_batched(model, device, global_windows, batch_size=batch_size)

    # Aggregate per file
    results: List[Dict[str, Any]] = []
    for file_idx, (path, num) in enumerate(file_infos):
        idxs = file_windows[file_idx]
        probs = [all_probs[i] for i in idxs]
        agg = aggregate_probs(probs, aggregate)
        if invert:
            agg = 1.0 - agg
        results.append({
            "file": str(path),
            "prob": float(agg),
            "num_windows": int(num),
        })
    return results


def parse_exts(exts_str: str) -> List[str]:
    parts = [e.strip().lower() for e in exts_str.split(",") if e.strip()]
    parts = [e if e.startswith(".") else f".{e}" for e in parts]
    return parts if parts else [".wav"]


def main():
    parser = argparse.ArgumentParser(description="Batch inference for raw-waveform 1D-CNN TorchScript model on multiple audio files.")
    parser.add_argument("--model", type=Path, required=True, help="Path to TorchScript model (.pt)")
    parser.add_argument("--wav_dir", type=Path, default=None, help="Directory to scan for audio files")
    parser.add_argument("--files", type=str, nargs="*", default=None, help="Explicit audio file paths and/or directories")
    parser.add_argument("--exts", type=str, default=".wav", help="Comma-separated list of extensions, e.g. .wav,.flac")

    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate expected by model")
    parser.add_argument("--window_s", type=float, default=1.0, help="Sliding window length in seconds")
    parser.add_argument("--hop_s", type=float, default=0.5, help="Sliding window hop in seconds")
    parser.add_argument("--pad", type=str, default="center", choices=["center", "right", "repeat"], help="Padding strategy for short clips")

    parser.add_argument("--norm", type=str, default="rms", choices=["rms", "peak", "none"], help="Normalization method")
    parser.add_argument("--no_dc", action="store_true", help="Disable DC removal prior to normalization")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for window inference")
    parser.add_argument("--aggregate", type=str, default="max", choices=["max", "mean", "median"], help="Aggregate window probabilities")
    parser.add_argument("--threshold", type=float, default=0.3, help="Decision threshold for positive class")
    parser.add_argument("--invert", action="store_true", help="Invert probability (1 - p) if model's positive class is reversed")

    parser.add_argument("--out_csv", type=Path, default=None, help="Optional: write per-file results to CSV")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    exts = parse_exts(args.exts)
    paths = collect_files(args.wav_dir, [Path(p) for p in (args.files or [])], exts)
    if not paths:
        print("No audio files found. Provide --wav_dir and/or --files.")
        sys.exit(2)

    model = load_torchscript(args.model, device)

    results = infer_many(
        model=model,
        device=device,
        paths=paths,
        target_sr=args.sr,
        window_s=args.window_s,
        hop_s=args.hop_s,
        pad=args.pad,
        norm=args.norm,
        remove_dc=(not args.no_dc),
        batch_size=args.batch_size,
        aggregate=args.aggregate,
        invert=args.invert,
    )

    # Print and threshold
    header = ["file", "prob", "pred", "num_windows"]
    rows = []
    for r in results:
        prob = float(r["prob"])
        pred = int(prob >= args.threshold)
        rows.append([r["file"], f"{prob:.6f}", pred, r["num_windows"]])
        print(f"{Path(r['file']).name}: prob={prob:.3f} pred={pred} windows={r['num_windows']}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()