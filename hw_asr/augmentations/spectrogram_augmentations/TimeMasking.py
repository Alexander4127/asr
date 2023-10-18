from torch import distributions, Tensor
import torchaudio

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, max_len: int = 20):
        assert 0 < max_len
        self._masking = torchaudio.transforms.TimeMasking(max_len)

    def __call__(self, data: Tensor) -> Tensor:
        return self._masking(data)
