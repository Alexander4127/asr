import random
from typing import Callable, Tuple, Optional

from torch import Tensor


class RandomApply:
    def __init__(self, augmentation: Optional[Callable], p: Optional[float]):
        assert p is None or 0 <= p <= 1
        self.augmentation = augmentation
        self.p = p

    def __call__(self, data: Tensor) -> Tuple[bool, Tensor]:
        if self.augmentation is None or random.random() > self.p:
            return False, data
        else:
            return True, self.augmentation(data)

    def __repr__(self):
        return repr(self.augmentation)
