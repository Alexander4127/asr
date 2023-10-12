import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}

    # spectrogram
    lengths = []
    for item in dataset_items:
        lengths.append(item['spectrogram'].shape[2])
    result_batch['spectrogram_length'] = torch.tensor(lengths)
    result_batch['spectrogram'] = torch.zeros([len(lengths), dataset_items[0]['spectrogram'].shape[1], max(lengths)])
    for idx, item in enumerate(dataset_items):
        result_batch['spectrogram'][idx, :, :lengths[idx]] = item['spectrogram']

    # text encoded
    lengths = []
    for item in dataset_items:
        lengths.append(item['text_encoded'].shape[1])
    result_batch['text_encoded_length'] = torch.tensor(lengths)
    result_batch['text_encoded'] = torch.zeros([len(lengths), max(lengths)])
    for idx, item in enumerate(dataset_items):
        result_batch['text_encoded'][idx, :lengths[idx]] = item['text_encoded']

    # texts
    result_batch['text'] = [item['text'] for item in dataset_items]

    # others
    for k in set(dataset_items[0].keys()) - set(result_batch.keys()):
        result_batch[k] = [item[k] for item in dataset_items]

    return result_batch
