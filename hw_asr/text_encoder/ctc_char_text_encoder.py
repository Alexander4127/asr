from typing import List, Dict, Any, Optional

import torch

from .ctc_text_encoder import CTCTextEncoder
from .lm import LMModel


class CTCCharTextEncoder(CTCTextEncoder):
    def __init__(self, alphabet: List[str] = None, lm_params: Optional[Dict[str, Any]] = None):
        super().__init__(alphabet)
        vocab: List[str] = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.lm_model: Optional[LMModel] = LMModel([''] + list(self.alphabet), **lm_params) \
            if lm_params is not None else None

    def model_beam_search(self, logits: torch.Tensor, probs_length, beam_size: int = 100):
        assert self.lm_model is not None
        return self.lm_model.decode_beams(logits, probs_length, beam_size)
