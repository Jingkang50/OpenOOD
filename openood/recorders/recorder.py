from openood.utils import Config


import torch.nn as nn


from numbers import Real
from typing import Protocol


class RecorderProtocol(Protocol):
    @property
    def output_dir(self) -> str:
        ...

    def report(self, train_metrics: dict[str, Real], val_metrics: dict[str, Real]):
        ...

    def save_model(self, net: nn.Module, val_metrics: dict[str, Real]):
        ...

    def summary(self):
        ...