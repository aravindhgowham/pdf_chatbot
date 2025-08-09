import torch
from typing import Tuple


def random_time_shift(waveforms: torch.Tensor, max_shift_samples: int) -> torch.Tensor:
    """
    Circularly shift each sample by a random amount up to +/- max_shift_samples.
    waveforms: [B, 1, T]
    """
    if max_shift_samples <= 0:
        return waveforms
    batch_size, _, T = waveforms.shape
    shifts = torch.randint(low=-max_shift_samples, high=max_shift_samples + 1, size=(batch_size,), device=waveforms.device)
    shifted = []
    for i in range(batch_size):
        s = int(shifts[i].item())
        shifted.append(torch.roll(waveforms[i], shifts=s, dims=-1))
    return torch.stack(shifted, dim=0)


def mixup_waveforms(waveforms: torch.Tensor, labels: torch.Tensor, num_classes: int, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mixup on waveforms and one-hot labels.
    Returns mixed_waveforms [B, 1, T] and soft_labels [B, C].
    """
    if alpha <= 0:
        one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        return waveforms, one_hot

    lam = torch.distributions.Beta(alpha, alpha).sample((waveforms.size(0),)).to(waveforms.device)
    lam = torch.maximum(lam, 1.0 - lam)  # encourage lambda >= 0.5
    index = torch.randperm(waveforms.size(0), device=waveforms.device)

    mixed = lam.view(-1, 1, 1) * waveforms + (1 - lam).view(-1, 1, 1) * waveforms[index]

    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    labels_mixed = lam.view(-1, 1) * labels_one_hot + (1 - lam).view(-1, 1) * labels_one_hot[index]
    return mixed, labels_mixed