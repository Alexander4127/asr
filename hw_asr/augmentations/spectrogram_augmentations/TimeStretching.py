import torchaudio
from torch import distributions, Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeStretching(AugmentationBase):
    def __init__(self, min_speed=0.5, max_speed=1.5):
        assert 0 < min_speed < max_speed, f'Min and max speed up must satisfy 0 < {min_speed} < {max_speed}'
        self._sampler = distributions.Uniform(low=min_speed, high=max_speed)
        self._stretch = torchaudio.transforms.TimeStretch(n_freq=128)

    def __call__(self, data: Tensor):
        factor = 1 / self._sampler.sample().item()
        return self._stretch(data, factor)
