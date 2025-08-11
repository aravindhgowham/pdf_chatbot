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
    target_sample_rate: int = 16000  # tag: target SR all files will be resampled to (Hz)
    duration_sec: float = 4.0        # tag: exact duration enforced per sample (seconds)
    random_crop: bool = True         # tag: if True and training, crop start is random; else center
    normalize: bool = True           # tag: if True, peak-normalize waveform to ~[-1, 1]


class RawAudioFolderDataset(Dataset):
    """
    Expects a directory with subfolders as class names: e.g.,
      root_dir/
        pass/*.wav
        fail/*.wav
    """

    def __init__(
        self,
        root_dir: str,                                  # tag: root folder containing class subfolders
        preproc: AudioPreprocConfig,                    # tag: preprocessing configuration
        class_name_to_index: Optional[Dict[str, int]] = None,  # tag: optional fixed label mapping
        augment: bool = False,                          # tag: whether to apply train-time augmentation
    ) -> None:
        super().__init__()
        self.root_dir = root_dir                        # tag: string path
        self.preproc = preproc                          # tag: AudioPreprocConfig instance in use
        self.augment = augment                          # tag: bool controlling augmentation

        # Discover classes by folder names if not provided
        if class_name_to_index is None:
            class_names = sorted(                       # tag: list[str] of subfolder names (classes)
                [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            )
            self.class_name_to_index = {name: idx for idx, name in enumerate(class_names)}  # tag: dict[class_name->int]
        else:
            self.class_name_to_index = class_name_to_index  # tag: use provided mapping

        self.filepaths: List[str] = []                  # tag: list of .wav file paths
        self.labels: List[int] = []                     # tag: parallel list of integer labels
        for class_name, class_idx in self.class_name_to_index.items():
            pattern = os.path.join(root_dir, class_name, "**", "*.wav")  # tag: glob pattern for wav files
            for fp in glob.glob(pattern, recursive=True):
                self.filepaths.append(fp)               # tag: append absolute/relative file path
                self.labels.append(class_idx)           # tag: append corresponding class index

        if len(self.filepaths) == 0:
            raise ValueError(f"No .wav files found under {root_dir}")  # tag: guard if dataset is empty

        self.num_samples = int(self.preproc.target_sample_rate * self.preproc.duration_sec)  # tag: fixed length T (samples)
        self.resampler_cache: Dict[int, T.Resample] = {}  # tag: cache of Resample transforms keyed by original SR

    def __len__(self) -> int:
        return len(self.filepaths)                      # tag: total number of audio examples

    def _get_resampler(self, orig_sr: int) -> Optional[T.Resample]:
        if orig_sr == self.preproc.target_sample_rate:
            return None                                 # tag: no-op if SR already matches target
        if orig_sr not in self.resampler_cache:
            self.resampler_cache[orig_sr] = T.Resample(orig_freq=orig_sr, new_freq=self.preproc.target_sample_rate)  # tag: create and cache resampler
        return self.resampler_cache[orig_sr]            # tag: return cached Resample transform

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [channels, time]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)            # tag: ensure shape is [1, time] if 1D
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # tag: average channels -> [1, time]
        return waveform                                 # tag: mono waveform tensor [1, time]

    def _pad_or_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [1, time]
        current_len = waveform.size(1)                  # tag: current number of samples (int)
        target_len = self.num_samples                   # tag: desired number of samples (int)
        if current_len == target_len:
            return waveform                             # tag: already correct length [1, target_len]
        if current_len < target_len:
            pad_len = target_len - current_len          # tag: how many zeros to append (int)
            pad_tensor = torch.zeros((1, pad_len), dtype=waveform.dtype)  # tag: [1, pad_len] zeros
            return torch.cat([waveform, pad_tensor], dim=1)  # tag: right-pad to [1, target_len]
        # crop
        if self.preproc.random_crop and self.augment:
            start = torch.randint(0, current_len - target_len + 1, (1,)).item()  # tag: random start index (int)
        else:
            start = max(0, (current_len - target_len) // 2)  # tag: center-crop start index (int)
        return waveform[:, start : start + target_len]       # tag: cropped waveform [1, target_len]

    def _maybe_augment(self, waveform: torch.Tensor) -> torch.Tensor:
        # Simple waveform-domain augmentations (noise removed by request)
        if not self.augment:
            return waveform                                 # tag: no change if augment=False
        # Random gain only (no additive noise)
        if torch.rand(1).item() < 0.5:
            gain = 0.9 + 0.2 * torch.rand(1).item()        # tag: scalar in [0.9, 1.1]
            waveform = waveform * gain                     # tag: scaled waveform [1, T]
        return waveform                                    # tag: augmented waveform [1, T]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath = self.filepaths[idx]                     # tag: string path to sample idx
        label = self.labels[idx]                           # tag: integer class index for sample idx

        waveform, sample_rate = torchaudio.load(filepath)  # tag: waveform [C, T_raw], sample_rate int
        waveform = self._to_mono(waveform)                 # tag: waveform -> [1, T_raw]

        resampler = self._get_resampler(sample_rate)       # tag: None or Resample transform
        if resampler is not None:
            waveform = resampler(waveform)                 # tag: resampled waveform [1, T_resampled]

        waveform = self._pad_or_crop(waveform)             # tag: waveform with exact target_len [1, T]
        waveform = self._maybe_augment(waveform)           # tag: maybe scaled waveform [1, T]

        if self.preproc.normalize:
            max_val = waveform.abs().max().clamp(min=1e-6) # tag: scalar peak amplitude > 0
            waveform = waveform / max_val                  # tag: peak-normalized waveform ~[-1, 1]

        return waveform, label                             # tag: (Tensor[1, T], int)

    def class_distribution(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}                        # tag: dict[label->count]
        for lbl in self.labels:
            counts[lbl] = counts.get(lbl, 0) + 1           # tag: increment per label
        return counts                                      # tag: return histogram of labels

    def get_label_mapping(self) -> Dict[str, int]:
        return dict(self.class_name_to_index)              # tag: dict[class_name->label_index]