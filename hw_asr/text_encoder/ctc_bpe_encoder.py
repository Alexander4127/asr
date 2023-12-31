import logging
import os
from typing import List, NamedTuple, Dict, Any, Optional
from string import ascii_lowercase
from pathlib import Path

import torch
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, Replace
from tokenizers import pre_tokenizers

from hw_asr.utils import ROOT_PATH
from .ctc_text_encoder import CTCTextEncoder


logger = logging.getLogger()


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCBPETextEncoder(CTCTextEncoder):
    EMPTY_TOK = "^"
    UNK_TOK = "<unk>"

    def __init__(self, vocab_size: int = 2000, use_pretrained: bool = True):
        super().__init__()
        self.tokenizer: Optional[Tokenizer] = None
        self.vocab_size = vocab_size
        self._tokenizer_path = str(ROOT_PATH / "hw_asr" / "text_encoder" / "tokenizer.json")
        if use_pretrained and os.path.exists(self._tokenizer_path):
            self.tokenizer = Tokenizer.from_file(self._tokenizer_path)
            self.vocab_size = self.tokenizer.get_vocab_size()
            if self.tokenizer.get_vocab_size() != vocab_size:
                logger.warning(f'Pretrained tokenizer has different vocab_size = {self.vocab_size}. Ignoring it.')
                self.tokenizer = None
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.tokenizer is not None:
            self.update_vocab(self.tokenizer.get_vocab())

    def encode(self, text) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text).ids).unsqueeze(0)

    def train_tokenizer(self, files: List[str]):
        initial_alphabet = [' '] + list(ascii_lowercase)
        self.tokenizer = Tokenizer(BPE(unk_token=self.UNK_TOK))
        trainer = BpeTrainer(
            special_tokens=[self.EMPTY_TOK, " "],
            initial_alphabet=initial_alphabet,
            vocab_size=self.vocab_size
        )

        self.tokenizer.normalizer = normalizers.Sequence([
            Lowercase(),
            Replace(Regex(r'[0-9]'), ''),
            Replace(Regex('-- |- |-|\n'), '')
        ])
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        self.tokenizer.train(files, trainer)
        self.tokenizer.save(self._tokenizer_path)
        self.update_vocab(self.tokenizer.get_vocab())
