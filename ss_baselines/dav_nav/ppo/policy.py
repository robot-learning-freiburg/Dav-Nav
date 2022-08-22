#!/usr/bin/env python3

# Edited by Abdelrahman Younes @ University of Freiburg
# Email: younesa@cs.uni-freiburg.de

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging

import torch
import torch.nn as nn
from torchsummary import summary

from ss_baselines.common.utils import CategoricalNetWithMask, AuxNet
from ss_baselines.common.visual_cnn import RNNStateEncoder
from ss_baselines.dav_nav.models.visual_cnn import VisualCNN
from ss_baselines.dav_nav.models.audio_autoencoder import AudioAutoEncoder
from ss_baselines.dav_nav.models.audio_visual_cnn import AudioVisualCNN
DUAL_GOAL_DELIMITER = ','

class Policy(nn.Module):
    def __init__(self, net, dim_actions, masking=True):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNetWithMask(
            self.net.output_size, self.dim_actions, masking
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
            self,
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=False,
    ):
        features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, distribution

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
            self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states, aux_loss = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, aux_loss


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class AudioNavBaselinePolicy(Policy):
    def __init__(
            self,
            observation_space,
            goal_sensor_uuid,
            masking,
            action_map_size,
            hidden_size=512,
            encode_rgb=False,
            encode_depth=False,
            use_aux_loss=False,
    ):
        super().__init__(
            AudioNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                encode_rgb=encode_rgb,
                encode_depth=encode_depth,
                use_aux_loss=use_aux_loss
            ),
            # action_space.n,
            action_map_size ** 2,
            masking=masking
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class AudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, encode_rgb, encode_depth, use_aux_loss):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._spectrogram = False
        self._gm = 'gm' in observation_space.spaces
        self._am = 'am' in observation_space.spaces

        self._spectrogram = 'spectrogram' == self.goal_sensor_uuid
        self._use_aux_loss = use_aux_loss
        self.aux_loss = torch.nn.MSELoss()

        #new architecture
        self.audi_visual_encoder = AudioVisualCNN(observation_space, hidden_size, map_type='gm')
        self.audio_auto_encoder = AudioAutoEncoder(observation_space, hidden_size)
        self.visual_encoder = VisualCNN(observation_space, hidden_size, encode_rgb, encode_depth)
        rnn_input_size = self._hidden_size * 3

        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if self._use_aux_loss:
            audio_encoder_output, upsampling_output, aux = self.audio_auto_encoder(observations)
        else:
            audio_encoder_output, upsampling_output = self.audio_auto_encoder(observations)
        x.append(audio_encoder_output)
        x.append(self.audi_visual_encoder(observations, upsampling_output))
        x.append(self.visual_encoder(observations))

        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()
        if self._use_aux_loss:
            if "filtered_spectrogram" in observations.keys():
                aux_loss = self.aux_loss(input=aux, target=observations['filtered_spectrogram'])
            else:
                aux_loss = self.aux_loss(input=aux, target=observations['spectrogram'])
        else:
            aux_loss = torch.zeros([1]).to(x2.device)

        return x2, rnn_hidden_states1, aux_loss
