#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn

from ss_baselines.common.utils import Flatten
from ss_baselines.common.visual_cnn import conv_output_dim, layer_init


class AudioAutoEncoder(nn.Module):
    r"""A Simple 3-Conv CNN for processing audio spectrogram

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size):
        super().__init__()
        self._n_input_audio = observation_space.spaces["spectrogram"].shape[2]

        cnn_dims = np.array(
            observation_space.spaces["spectrogram"].shape[:2], dtype=np.float32
        )

        if cnn_dims[0] < 30 or cnn_dims[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
            self.decoder = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=(5, 6), stride=(2, 2)),
                nn.ReLU(True),
            )
            self.upsampling = nn.Sequential(
                nn.ConvTranspose2d(in_channels=2, out_channels=32, kernel_size=(5, 2), stride=(3, 4)),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(4, 2), stride=(1, 2)),
                nn.ReLU(True),
                nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 5), stride=(1, 1)),
                nn.ReLU(True),
            )
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]
            self.decoder = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(1, 1)),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(10, 10), stride=(1, 1)),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=(5, 5), stride=(4, 4)),
                nn.ReLU(True),
            )
            self.upsampling = nn.Sequential(
                nn.ConvTranspose2d(in_channels=2, out_channels=32, kernel_size=(8, 8), stride=(3, 3)),
                nn.ReLU(True),
                nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 13), stride=(1, 1)),
                nn.ReLU(True),
            )

        for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels= self._n_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            # nn.ReLU(True),
        )

        self.cnn = nn.Sequential(
            Flatten(),
            nn.Linear(32 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )

        layer_init(self.encoder)
        layer_init(self.decoder)
        layer_init(self.cnn)

    def forward(self, observations):
        cnn_input = []

        audio_observations = observations["spectrogram"]
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        audio_observations = audio_observations.permute(0, 3, 1, 2)
        cnn_input.append(audio_observations)

        cnn_input = torch.cat(cnn_input, dim=1)
        # encoder_output = self.encoder((cnn_input[:, 0, :, :] - cnn_input[:, 1, :, :]).unsqueeze(1))
        encoder_output = self.encoder(cnn_input)
        # decoder_output = self.decoder(encoder_output)
        upsampling_output = self.upsampling(cnn_input)
        return self.cnn(encoder_output) , upsampling_output # , decoder_output.permute(0, 2, 3, 1)
