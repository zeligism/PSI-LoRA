import json
from typing import Any, Optional
from collections import defaultdict
from pathlib import Path
import random

import torch
import torch.nn as nn
import numpy as np


def init_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def init_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Metrics:
    """
    This utility class stores the metrics per epoch.

    The update takes a dict of values and append them to self.
    We make an index mandatory instead of appending to a list
    (which is helpful for continuing interrupted runs).
    The get is kinda useless.

    It also has helper methods for load and save as json.
    """
    def __init__(self):
        # metrics: dict[metric_name, dict[index, metric_val]]
        # inner val is not list[metric_val], mainly for convenience
        self.metrics = defaultdict(dict)

    def update(self, index: float, metrics: dict[str, Any]):
        for metric_name, metric_val in metrics.items():
            self.metrics[metric_name][index] = metric_val

    def get(self, index : float, metric_name: str) -> list[Any]:
        if index == -1:
            index = max(self.metrics[metric_name].keys())
        return self.metrics[metric_name][index]

    def get_last_index(self, metric_name: Optional[str] = None) -> float:
        # TODO: rename to get_last_epoch or something
        if len(self.metrics) == 0:
            return 0
        metric_name = metric_name or next(iter(self.metrics.keys()))
        return max(map(int, self.metrics[metric_name].keys()))

    def get_last_metric(self, metric_name: str) -> float:
        assert metric_name in self.metrics
        last = str(self.get_last_index(metric_name))
        return self.metrics[metric_name][last]

    def save(self, fname: Path):
        with open(fname, "w") as f:
            json.dump(self.metrics, f, indent=4)

    def load(self, fname: Path):
        with open(fname, "r") as f:
            self.metrics = json.load(f)

    def to_pandas(self):
        from pandas import DataFrame
        return DataFrame(self.metrics).reset_index(names="epochs")


class AverageMeters:
    """
    Just an easier way to track averages from a dict of values.
    """
    def __init__(self):
        self.counts = defaultdict(int)
        self.sums = defaultdict(float)

    def update(self, metrics: dict[str, float]):
        for key, val in metrics.items():
            self.sums[key] += val
            self.counts[key] += 1

    def get(self, key: str) -> float:
        if self.counts[key] > 0:
            return self.sums[key] / self.counts[key]
        else:
            return 0.

    def asdict(self, prefix: str = "") -> dict[str, float]:
        return {prefix + key: self.get(key) for key in self.counts.keys()}
    

@torch.no_grad()
def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    _, predicted = outputs.max(1)
    return predicted.eq(labels).float().mean().item()


class ExcessMSELoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction = "mean", min_loss: float = 0.0):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.min_loss = min_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target) - self.min_loss
