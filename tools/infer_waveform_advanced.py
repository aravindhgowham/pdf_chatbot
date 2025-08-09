#!/usr/bin/env python3
from pathlib import Path
from typing import List, Tuple, Dict, Any

import math
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
# Audio utils
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
    g = math.gcd(src_sr, dst_sr)
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


def chunk_audio(audio: np.ndarray, sr: int, window_samples: int, hop_samples: int, pad: str = "center") -> List[np.ndarray]:
    if window_samples <= 0:
        return []
    chunks: List[np.ndarray] = []
    if len(audio) < window_samples:
        pad_len = window_samples - len(audio)
        if pad == "center":
            left = pad_len // 2
            right = pad_len - left
            padded = np.pad(audio, (left, right))
        elif pad == "right":
            padded = np.pad(audio, (0, pad_len))
        elif pad == "repeat":
            reps = int(np.ceil(window_samples / max(1, len(audio))))
            padded = np.tile(audio, reps)[:window_samples]
        else:
            raise ValueError(f"Unknown pad: {pad}")
        chunks.append(padded.astype(np.float32))
        return chunks
    if hop_samples <= 0:
        hop_samples = window_samples
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        end = start + window_samples
        chunks.append(audio[start:end].astype(np.float32))
    return chunks


# =========================
# Inference helpers
# =========================

def load_torchscript(model_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def apply_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        return prob
    # Convert to logits, apply temperature, back to probability
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    logit = np.log(prob / (1 - prob))
    scaled = logit / temperature
    prob2 = 1.0 / (1.0 + np.exp(-scaled))
    return prob2.astype(np.float32)


def tta_time_shifts(windows: List[np.ndarray], shifts: List[int]) -> List[np.ndarray]:
    if not shifts:
        return windows
    aug: List[np.ndarray] = []
    for w in windows:
        aug.append(w)
        for s in shifts:
            if s == 0:
                continue
            if s > 0:
                pad = np.zeros(s, dtype=np.float32)
                aug.append(np.concatenate([pad, w[:-s]]))
            else:
                s2 = -s
                pad = np.zeros(s2, dtype=np.float32)
                aug.append(np.concatenate([w[s2:], pad]))
    return aug


def infer_windows_batched(model, device: torch.device, windows: List[np.ndarray], batch_size: int) -> List[float]:
    probs: List[float] = []
    total = len(windows)
    rng = range(0, total, batch_size)
    if _HAVE_TQDM:
        rng = tqdm(rng, total=(total + batch_size - 1) // batch_size, desc="Batches")
    with torch.no_grad():
        for i in rng:
            batch = windows[i:i + batch_size]
            x = np.stack(batch, axis=0)
            xt = torch.from_numpy(x).to(device).float()[:, None, :]
            y = model(xt)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            if y.shape[-1] == 1:
                p = torch.sigmoid(y).squeeze(-1)
            else:
                p = torch.softmax(y, dim=-1)[..., -1]
            probs.extend(p.detach().cpu().numpy().astype(np.float32).tolist())
    return probs


def aggregate_probs(probs: List[float], mode: str, top_k: int = 0) -> float:
    if not probs:
        return 0.0
    arr = np.array(probs, dtype=np.float32)
    if top_k and top_k > 0:
        k = min(top_k, arr.size)
        arr = np.partition(arr, -k)[-k:]
    if mode == "max":
        return float(np.max(arr))
    if mode == "mean":
        return float(np.mean(arr))
    if mode == "median":
        return float(np.median(arr))
    raise ValueError(f"Unknown aggregate: {mode}")


def build_windows_multiscale(audio: np.ndarray, sr: int, scales_s: List[float], hop_fraction: float, pad: str) -> List[np.ndarray]:
    all_windows: List[np.ndarray] = []
    for ws in scales_s:
        win = int(ws * sr)
        hop = max(1, int(ws * hop_fraction * sr))
        all_windows.extend(chunk_audio(audio, sr, win, hop, pad=pad))
    return all_windows


def infer_files(
    model,
    device: torch.device,
    files: List[Path],
    sr: int,
    norm: str,
    remove_dc: bool,
    scales_s: List[float],
    hop_fraction: float,
    pad: str,
    batch_size: int,
    tta_shifts: List[int],
    aggregate: str,
    top_k: int,
    invert: bool,
    temperature: float,
) -> List[Dict[str, Any]]:

    # Build all windows across files with multi-scale and TTA shifts
    global_windows: List[np.ndarray] = []
    file_spans: List[Tuple[int, int]] = []

    file_iter = files
    if _HAVE_TQDM:
        file_iter = tqdm(files, desc="Reading")

    for f in file_iter:
        audio, src_sr = read_audio(f)
        audio = to_mono(audio)
        audio = resample(audio, src_sr, sr)
        audio = normalize_audio(audio, method=norm, remove_dc=remove_dc)
        windows = build_windows_multiscale(audio, sr, scales_s, hop_fraction, pad)
        windows = tta_time_shifts(windows, tta_shifts)
        start = len(global_windows)
        global_windows.extend(windows)
        end = len(global_windows)
        file_spans.append((start, end))

    # Run batched inference
    probs = infer_windows_batched(model, device, global_windows, batch_size=batch_size)

    # Aggregate per file
    results: List[Dict[str, Any]] = []
    for (start, end), f in zip(file_spans, files):
        p = np.array(probs[start:end], dtype=np.float32)
        if p.size == 0:
            agg = 0.0
        else:
            p = apply_temperature(p, temperature)
            agg = aggregate_probs(p.tolist(), aggregate, top_k=top_k)
        if invert:
            agg = 1.0 - agg
        results.append({"file": str(f), "prob": float(agg), "num_windows": int(max(0, end - start))})
    return results


def main():
    # =========================
    # Configuration (edit here)
    # =========================
    config = {
        # Paths
        "model": Path("/abs/path/to/model.pt"),
        "inputs": [
            Path("/abs/path/to/audio_dir_or_file1.wav"),
            Path("/abs/path/to/audio_dir_or_file2.wav"),
        ],
        "exts": [".wav"],  # used only for directories

        # Audio/model expectations
        "sr": 16000,
        "norm": "rms",          # rms | peak | none
        "remove_dc": True,

        # Multi-scale windows
        "scales_s": [0.5, 1.0, 1.5],   # seconds
        "hop_fraction": 0.5,           # hop = scale * hop_fraction
        "pad": "center",              # center | right | repeat

        # Inference
        "batch_size": 64,
        "tta_shifts": [0, 160, -160],  # samples; small shifts for robustness
        "aggregate": "max",            # max | mean | median
        "top_k": 5,                    # use top-k windows before aggregation; set 0 to disable
        "invert": False,               # set True if model polarity reversed
        "temperature": 1.0,            # >1.0 smooths, <1.0 sharpens probabilities

        # Output
        "threshold": 0.30,
        "out_csv": Path("/workspace/results_advanced.csv"),
    }

    # Expand input paths into files
    files: List[Path] = []
    for p in config["inputs"]:
        p = Path(p)
        if p.is_dir():
            for q in sorted(p.rglob("*")):
                if q.is_file() and q.suffix.lower() in config["exts"]:
                    files.append(q)
        elif p.is_file():
            files.append(p)
    if not files:
        print("No input audio files found. Update config['inputs'].")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_torchscript(config["model"], device)

    results = infer_files(
        model=model,
        device=device,
        files=files,
        sr=config["sr"],
        norm=config["norm"],
        remove_dc=config["remove_dc"],
        scales_s=config["scales_s"],
        hop_fraction=config["hop_fraction"],
        pad=config["pad"],
        batch_size=config["batch_size"],
        tta_shifts=config["tta_shifts"],
        aggregate=config["aggregate"],
        top_k=config["top_k"],
        invert=config["invert"],
        temperature=config["temperature"],
    )

    # Print and save
    rows = []
    for r in results:
        prob = float(r["prob"])
        pred = int(prob >= config["threshold"])
        print(f"{Path(r['file']).name}: prob={prob:.3f} pred={pred} windows={r['num_windows']}")
        rows.append([r["file"], f"{prob:.6f}", pred, r["num_windows"]])

    out_csv: Path = config["out_csv"]
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file", "prob", "pred", "num_windows"])
            writer.writerows(rows)
        print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()