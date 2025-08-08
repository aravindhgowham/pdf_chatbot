import argparse
import os
import torch
import torch.nn.functional as F
import torchaudio
from dotenv import load_dotenv

from dataset import AudioPreprocConfig
from model import RawAudioCNN1D
from utils import load_yaml


# Load environment variables from .env if present
load_dotenv()
ENV_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
ENV_DURATION_SEC = float(os.getenv("AUDIO_DURATION_SEC", "4.0"))


def load_and_prepare(audio_path: str, preproc: AudioPreprocConfig) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_path)  # [C, T]

    # Mono
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != preproc.target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=preproc.target_sample_rate)
        waveform = resampler(waveform)

    # Pad/center-crop
    target_len = int(preproc.target_sample_rate * preproc.duration_sec)
    current_len = waveform.size(1)
    if current_len < target_len:
        pad = torch.zeros((1, target_len - current_len), dtype=waveform.dtype)
        waveform = torch.cat([waveform, pad], dim=1)
    elif current_len > target_len:
        start = max(0, (current_len - target_len) // 2)
        waveform = waveform[:, start : start + target_len]

    # Normalize peak
    max_val = waveform.abs().max().clamp(min=1e-6)
    waveform = waveform / max_val
    return waveform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for raw audio PASS/FAIL classifier")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--label_map", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=ENV_SAMPLE_RATE)
    parser.add_argument("--duration_sec", type=float, default=ENV_DURATION_SEC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    label_map = load_yaml(args.label_map)
    index_to_label = {v: k for k, v in label_map.items()}

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model = RawAudioCNN1D(in_channels=1, num_classes=len(index_to_label))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preproc = AudioPreprocConfig(target_sample_rate=args.sample_rate, duration_sec=args.duration_sec)
    waveform = load_and_prepare(args.audio_path, preproc)
    with torch.no_grad():
        logits = model(waveform.unsqueeze(0))  # [1, 1, T]
        probs = F.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        pred_label = index_to_label[pred_idx]
        pred_prob = float(probs[pred_idx].item())

    print(f"Prediction: {pred_label.upper()} (p={pred_prob:.2f})")


if __name__ == "__main__":
    main()