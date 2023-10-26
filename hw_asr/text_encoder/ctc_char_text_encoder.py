import logging
from typing import List, Dict, Any, Optional

import torch

from .ctc_text_encoder import CTCTextEncoder
from .lm import LMModel


logger = logging.getLogger()


class CTCCharTextEncoder(CTCTextEncoder):
    def __init__(self, alphabet: List[str] = None, lm_params: Optional[Dict[str, Any]] = None):
        super().__init__(alphabet)
        vocab: List[str] = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.lm_model: Optional[LMModel] = LMModel([''] + list(self.alphabet), **lm_params) \
            if lm_params is not None else None

    def model_beam_search(self, logits: torch.Tensor, probs_length, pool, beam_size: int = 100):
        if self.lm_model is None:
            logger.warning("Cannot use LM. To remove this warning delete model_beam_search call from test.py")
            return [""] * len(logits)
        return self.lm_model.decode_beams(logits, probs_length, pool, beam_size)
