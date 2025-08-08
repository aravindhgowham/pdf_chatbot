import os
import random
import yaml
from typing import Dict, List

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_yaml(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_class_weights(counts: Dict[int, int]) -> torch.Tensor:
    # inverse frequency
    total = sum(counts.values())
    weights: List[float] = []
    for cls_idx in sorted(counts.keys()):
        freq = counts[cls_idx] / total
        weights.append(1.0 / (freq + 1e-8))
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    weights_tensor = weights_tensor / weights_tensor.sum() * len(weights)
    return weights_tensor