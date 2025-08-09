#!/usr/bin/env python3
import argparse
import os
import sys
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.signal import resample_poly
try:
    import soundfile as sf
    _HAVE_SF = True
except Exception:
    _HAVE_SF = False
    from scipy.io import wavfile


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    if _HAVE_SF:
        audio, sr = sf.read(str(path), always_2d=False)
        # Convert to float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        return audio, sr
    else:
        sr, audio = wavfile.read(str(path))
        # Convert to float32 and scale if necessary
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
    # Mix down channels equally
    return np.mean(audio, axis=1).astype(np.float32)


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    # Use polyphase resampling for quality
    g = np.gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    return resample_poly(audio, up=up, down=down).astype(np.float32)


def remove_dc_and_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    audio = audio - np.mean(audio)
    rms = np.sqrt(np.mean(np.square(audio)) + eps)
    if rms > 0:
        audio = audio / rms
    return audio.astype(np.float32)


def chunk_audio(audio: np.ndarray, sr: int, window_seconds: float, hop_seconds: float) -> List[np.ndarray]:
    win = int(window_seconds * sr)
    hop = int(hop_seconds * sr)
    if win <= 0:
        return []
    chunks: List[np.ndarray] = []
    if len(audio) < win:
        # Pad symmetrically
        pad = win - len(audio)
        left = pad // 2
        right = pad - left
        padded = np.pad(audio, (left, right))
        chunks.append(padded)
        return chunks
    for start in range(0, len(audio) - win + 1, hop):
        end = start + win
        chunks.append(audio[start:end])
    return chunks


def load_torchscript(model_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def infer_file(model, device: torch.device, path: Path, target_sr: int, window_s: float, hop_s: float, aggregate: str) -> Tuple[float, List[float]]:
    audio, sr = read_audio(path)
    audio = to_mono(audio)
    audio = resample(audio, sr, target_sr)
    audio = remove_dc_and_normalize(audio)
    chunks = chunk_audio(audio, target_sr, window_s, hop_s)
    if not chunks:
        return 0.0, []

    with torch.no_grad():
        probs: List[float] = []
        for chunk in chunks:
            x = torch.from_numpy(chunk).to(device).float()[None, None, :]
            y = model(x)
            # Support either logits or probabilities
            if y.shape[-1] == 1:
                y = y.view(-1)
                p = torch.sigmoid(y).item()
            else:
                # Assume binary softmax [neg, pos]
                y = y.view(-1)
                p = torch.softmax(y, dim=0)[-1].item()
            probs.append(float(p))

    if aggregate == "max":
        agg = float(np.max(probs))
    elif aggregate == "mean":
        agg = float(np.mean(probs))
    elif aggregate == "median":
        agg = float(np.median(probs))
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")
    return agg, probs


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw-waveform 1D-CNN TorchScript model on a directory of WAV files.")
    parser.add_argument("--model", type=Path, required=True, help="Path to TorchScript model (.pt)")
    parser.add_argument("--wav_dir", type=Path, required=True, help="Directory of WAV files to evaluate")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate expected by model")
    parser.add_argument("--window_s", type=float, default=1.0, help="Sliding window length in seconds")
    parser.add_argument("--hop_s", type=float, default=0.5, help="Sliding window hop in seconds")
    parser.add_argument("--aggregate", type=str, default="max", choices=["max", "mean", "median"], help="Aggregate window probabilities")
    parser.add_argument("--threshold", type=float, default=0.3, help="Decision threshold for positive class")
    parser.add_argument("--ext", type=str, default=".wav", help="Audio file extension to scan")
    parser.add_argument("--out_csv", type=Path, default=None, help="Optional: write results to CSV")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--invert", action="store_true", help="Invert probability (1 - p) if model's positive class is reversed")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_torchscript(args.model, device)

    files = [p for p in sorted(args.wav_dir.rglob("*")) if p.is_file() and p.suffix.lower() == args.ext.lower()]
    if not files:
        print(f"No {args.ext} files found under {args.wav_dir}")
        sys.exit(2)

    header = ["file", "prob", "pred", "num_windows"]
    rows = []

    tp = fp = tn = fn = 0

    for f in files:
        prob, probs = infer_file(model, device, f, args.sr, args.window_s, args.hop_s, args.aggregate)
        if args.invert:
            prob = 1.0 - prob
        pred = int(prob >= args.threshold)
        rows.append([str(f), f"{prob:.6f}", pred, len(probs)])
        print(f"{f.name}: prob={prob:.3f} pred={pred} windows={len(probs)}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()