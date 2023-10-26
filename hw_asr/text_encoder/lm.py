import logging
import os
import shutil
from typing import NamedTuple

import wget
from pathlib import Path
import gzip

import torch
from pyctcdecode import build_ctcdecoder


ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent

VOCAB_DIR = ROOT_PATH / "data" / "datasets" / "librispeech" / "vocab"
VOCAB_LINK = "https://www.openslr.org/resources/11/librispeech-vocab.txt"
MODEL_DIR = ROOT_PATH / "hw_asr" / "text_encoder"
MODEL_LINK_PRUNED = "http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
MODEL_LINK = "http://www.openslr.org/resources/11/3-gram.arpa.gz"


logger = logging.getLogger()


class LMHypot(NamedTuple):
    text: str
    score: float
    lm_score: float


class LMModel:
    def __init__(self,
                 alphabet,
                 vocab_dir: Path = VOCAB_DIR,
                 vocab_link: str = VOCAB_LINK,
                 model_dir: Path = MODEL_DIR,
                 is_pruned: bool = False):
        self._dir: Path = vocab_dir
        self._model_dir: Path = model_dir
        self._filename = self._load_vocab(vocab_link)
        self._model_filename = self._load_model(MODEL_LINK_PRUNED if is_pruned else MODEL_LINK)

        unigrams = self._get_unigrams()
        ken_lm_path = str(self._model_dir / self._model_filename)
        self._decoder = build_ctcdecoder(alphabet, unigrams=unigrams, kenlm_model_path=ken_lm_path)

    def _load_vocab(self, load_link: str):
        os.makedirs(self._dir, exist_ok=True)
        filename = load_link.split('/')[-1].strip()
        if os.path.exists(self._dir / filename):
            return filename
        wget.download(load_link, out=str(self._dir))
        return filename

    def _load_model(self, load_link: str):
        os.makedirs(self._model_dir, exist_ok=True)
        zip_filename = load_link.split('/')[-1].strip()
        filename = zip_filename[:-3]
        assert zip_filename.endswith('.gz')

        lower_filename = 'lower_' + filename
        if os.path.exists(self._model_dir / lower_filename):
            return lower_filename

        if os.path.exists(self._model_dir / filename):
            return self._model_to_lower(filename, lower_filename)

        if not os.path.exists(self._model_dir / zip_filename):
            wget.download(load_link, out=str(self._model_dir))

        with gzip.open(self._model_dir / zip_filename, 'rb') as f_zipped:
            with open(self._model_dir / filename, 'wb') as f_unzipped:
                shutil.copyfileobj(f_zipped, f_unzipped)

        return self._model_to_lower(filename, lower_filename)

    def _model_to_lower(self, upper_filename, model_filename):
        with open(self._model_dir / upper_filename, 'r') as upper_model_file:
            with open(self._model_dir / model_filename, 'w') as model_file:
                for line in upper_model_file:
                    print(line.lower(), file=model_file, end='')
        return model_filename

    def _get_unigrams(self):
        assert os.path.exists(self._dir)
        unigrams = []
        with open(self._dir / self._filename) as file:
            for line in file.readlines():
                unigrams.append(line.strip().lower())
        logger.info(f'{unigrams[0], unigrams[-1]}')
        return unigrams

    def decode_beams(self, logits: torch.Tensor, probs_lengths, pool, beam_size: int):
        batch_probs = [logit[:length].cpu().numpy() for logit, length in zip(logits, probs_lengths)]
        return self._decoder.decode_batch(pool, batch_probs, beam_width=beam_size)
