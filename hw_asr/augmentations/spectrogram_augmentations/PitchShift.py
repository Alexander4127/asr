from torch_audiomentations import PitchShift
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, sample_rate):
        self._sr = sample_rate
        self._aug = PitchShift()

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x, sample_rate=self._sr).squeeze(1)
