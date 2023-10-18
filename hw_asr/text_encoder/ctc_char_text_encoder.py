import logging
from typing import List, NamedTuple, Dict, Any, Tuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab: List[str] = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        result = []
        empty_ind = self.char2ind[self.EMPTY_TOK]
        last_char = self.EMPTY_TOK
        for ind in inds:
            cur_char = self.ind2char[ind]
            if ind == empty_ind:
                last_char = cur_char
                continue
            if last_char != cur_char:
                result.append(cur_char)
            last_char = cur_char
        return ''.join(result)

    def _extend_and_merge(self, frame: torch.Tensor, state: Dict[Tuple[str, str], float]):
        assert len(frame.shape) == 1
        new_state = defaultdict(float)
        for next_char_idx, next_char_prob in enumerate(frame):
            for (pref, last_char), pref_prob in state.items():
                next_char = self.ind2char[next_char_idx]
                new_pref = pref if next_char == last_char or next_char == self.EMPTY_TOK else pref + next_char
                new_state[(new_pref, next_char)] += pref_prob * next_char_prob
        return dict(new_state)

    @staticmethod
    def _truncate(state, beam_size) -> Dict[Tuple[str, str], float]:
        state_list = sorted(state.items(), key=lambda x: -x[1])
        return dict(state_list[:beam_size])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        probs = probs[:probs_length]
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        state: Dict[Tuple[str, str], float] = {('', self.EMPTY_TOK): 1.0}
        for frame in probs:
            state = self._extend_and_merge(frame, state)
            state = self._truncate(state, beam_size)
        hypos: List[Hypothesis] = [Hypothesis(k[0], v) for k, v in state.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
