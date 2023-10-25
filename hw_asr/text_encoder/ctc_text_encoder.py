from collections import defaultdict
import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, NamedTuple, Dict, Any, Tuple, Optional, Union

import numpy as np
import torch

from hw_asr.base.base_text_encoder import BaseTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCTextEncoder(BaseTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        self.alphabet: Optional[List[str]] = None
        self.char2ind: Optional[Dict[str, int]] = None
        self.ind2char: Optional[Dict[int, str]] = None
        if alphabet is None:
            alphabet = list(ascii_lowercase + ' ')
        self.update_alphabet(alphabet)

    def update_alphabet(self, alphabet: List[str]):
        self.alphabet = alphabet
        self.ind2char = {k: v for k, v in enumerate(alphabet)}
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.ind2char)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError as e:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'")

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
        hypos: List[Hypothesis] = [Hypothesis(k[0], float(v)) for k, v in state.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def decode(self, vector: Union[torch.Tensor, np.ndarray, List[int]]):
        return ''.join([self.ind2char[int(ind)] for ind in vector]).strip()

    def dump(self, file):
        with Path(file).open('w') as f:
            json.dump(self.ind2char, f)

    @classmethod
    def from_file(cls, file):
        with Path(file).open() as f:
            ind2char = json.load(f)
        a = cls([])
        a.ind2char = ind2char
        a.char2ind = {v: k for k, v in ind2char}
        return a
