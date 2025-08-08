import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T


@dataclass
class AudioPreprocConfig:
    target_sample_rate: int = 16000
    duration_sec: float = 4.0
    random_crop: bool = True
    normalize: bool = True


class RawAudioFolderDataset(Dataset):
    """
    Expects a directory with subfolders as class names: e.g.,
      root_dir/
        pass/*.wav
        fail/*.wav
    """

    def __init__(
        self,
        root_dir: str,
        preproc: AudioPreprocConfig,
        class_name_to_index: Optional[Dict[str, int]] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.preproc = preproc
        self.augment = augment

        # Discover classes by folder names if not provided
        if class_name_to_index is None:
            class_names = sorted(
                [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            )
            self.class_name_to_index = {name: idx for idx, name in enumerate(class_names)}
        else:
            self.class_name_to_index = class_name_to_index

        self.filepaths: List[str] = []
        self.labels: List[int] = []
        for class_name, class_idx in self.class_name_to_index.items():
            pattern = os.path.join(root_dir, class_name, "**", "*.wav")
            for fp in glob.glob(pattern, recursive=True):
                self.filepaths.append(fp)
                self.labels.append(class_idx)

        if len(self.filepaths) == 0:
            raise ValueError(f"No .wav files found under {root_dir}")

        self.num_samples = int(self.preproc.target_sample_rate * self.preproc.duration_sec)
        self.resampler_cache: Dict[int, T.Resample] = {}

    def __len__(self) -> int:
        return len(self.filepaths)

    def _get_resampler(self, orig_sr: int) -> Optional[T.Resample]:
        if orig_sr == self.preproc.target_sample_rate:
            return None
        if orig_sr not in self.resampler_cache:
            self.resampler_cache[orig_sr] = T.Resample(orig_freq=orig_sr, new_freq=self.preproc.target_sample_rate)
        return self.resampler_cache[orig_sr]

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [channels, time]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def _pad_or_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [1, time]
        current_len = waveform.size(1)
        target_len = self.num_samples
        if current_len == target_len:
            return waveform
        if current_len < target_len:
            pad_len = target_len - current_len
            pad_tensor = torch.zeros((1, pad_len), dtype=waveform.dtype)
            return torch.cat([waveform, pad_tensor], dim=1)
        # crop
        if self.preproc.random_crop and self.augment:
            start = torch.randint(0, current_len - target_len + 1, (1,)).item()
        else:
            start = max(0, (current_len - target_len) // 2)
        return waveform[:, start : start + target_len]

    def _maybe_augment(self, waveform: torch.Tensor) -> torch.Tensor:
        # Simple waveform-domain augmentations (noise removed by request)
        if not self.augment:
            return waveform
        # Random gain only (no additive noise)
        if torch.rand(1).item() < 0.5:
            gain = 0.9 + 0.2 * torch.rand(1).item()  # [0.9, 1.1]
            waveform = waveform * gain
        return waveform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(filepath)  # [channels, time]
        waveform = self._to_mono(waveform)

        resampler = self._get_resampler(sample_rate)
        if resampler is not None:
            waveform = resampler(waveform)

        waveform = self._pad_or_crop(waveform)
        waveform = self._maybe_augment(waveform)

        if self.preproc.normalize:
            max_val = waveform.abs().max().clamp(min=1e-6)
            waveform = waveform / max_val

        return waveform, label

    def class_distribution(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for lbl in self.labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def get_label_mapping(self) -> Dict[str, int]:
        return dict(self.class_name_to_index)