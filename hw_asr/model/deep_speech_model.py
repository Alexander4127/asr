import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

from hw_asr.base import BaseModel


class RNNLayer(nn.Module):
    def __init__(self, n_feat, rnn_hid, rnn=nn.LSTM, bi=True, bn=True, **kwargs):
        super().__init__()
        self.n_feat = n_feat
        self.rnn_hid = rnn_hid
        self.bidirectional = bi
        self.relu = nn.ReLU()
        self.rnn = rnn(input_size=n_feat, hidden_size=rnn_hid, bidirectional=bi, batch_first=True)  # (B, T, H)
        self.batch_norm = nn.BatchNorm1d(num_features=n_feat) if bn else None  # (B, H, T)

    def forward(self, x, lengths, hid=None):
        assert len(x.shape) == 3 and x.shape[2] == self.n_feat, f'{x.shape}[2] != {self.n_feat}'  # (B, T, H)
        if self.batch_norm is not None:
            x = self.relu(self.batch_norm(x.transpose(1, 2)).transpose(1, 2))  # (B, T, H)
        b, t, h = x.shape

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, hid = self.rnn(x, hid)
        x, _ = pad_packed_sequence(x, batch_first=True)

        assert x.shape == torch.Size([b, t, (1 + self.bidirectional) * self.rnn_hid])
        return x


class DeepSpeechModel(BaseModel):
    def __init__(self, n_feats, n_class, rnn_params, conv_params, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.n_feats = n_feats
        self.n_class = n_class
        self.conv_params = conv_params

        # convs
        convs = []
        rnn_feat = n_feats
        for conv_layer in self.conv_params:
            assert rnn_feat % conv_layer['stride'][0] == 0
            rnn_feat = rnn_feat // conv_layer['stride'][0]
            convs.append(self._make_conv(**conv_layer))
        self.conv = nn.Sequential(*convs)

        # downsample to 1 channel
        self.downsample = nn.Conv2d(self.conv_params[-1]['out_channels'], 1, kernel_size=1)
        self.normalize = nn.Sequential(
            nn.BatchNorm1d(num_features=rnn_feat),
            nn.Hardtanh(min_val=0.0, max_val=20.0)
        )

        # rnn
        rnn_hid = (1 + rnn_params['bi']) * rnn_params['rnn_hid']
        self.rnn = nn.ModuleList([RNNLayer(n_feat=rnn_feat, **rnn_params)] +
                                 [RNNLayer(n_feat=rnn_hid, **rnn_params) for _ in range(rnn_params['num_rnn'] - 1)])

        # fc
        self.fc = nn.Linear(in_features=rnn_hid, out_features=n_class)

    @staticmethod
    def _make_conv(out_channels, kernel_size, **kwargs):
        padding = tuple(np.array(kernel_size) // 2)
        return nn.Sequential(
            nn.Conv2d(out_channels=out_channels, kernel_size=kernel_size, padding=padding, **kwargs, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Hardtanh(min_val=0.0, max_val=20)
        )

    def forward(self, spectrogram, spectrogram_length,  **batch):
        x = spectrogram  # (B, H, T)
        # logger = logging.getLogger()
        # logger.info(f'spec.shape = {spectrogram.shape}')
        assert len(x.shape) == 3 and x.shape[1] == self.n_feats, f'{x.shape}[2] != {self.n_feats}'

        x = self.conv(x.unsqueeze(1))
        x = self.downsample(x).squeeze(1)
        x = self.normalize(x)
        assert len(x.shape) == 3, f'After conv get {x.shape}'

        x = x.transpose(1, 2)  # (B, T, H)
        # logger.info(f'x after conv shape = {x.shape}')
        lengths = self.transform_input_lengths(spectrogram_length)
        for rnn_layer in self.rnn:
            x = rnn_layer(x, lengths)

        # logger.info(f'x after rnn shape = {x.shape}')
        return {"logits": self.fc(x)}

    def transform_input_lengths(self, input_lengths):
        for conv_layer in self.conv_params:
            assert 'dilation' not in conv_layer or conv_layer['dilation'] == 1
            input_lengths = (input_lengths.subtract(1).float() / conv_layer['stride'][1]).add(1).int()
        return input_lengths
