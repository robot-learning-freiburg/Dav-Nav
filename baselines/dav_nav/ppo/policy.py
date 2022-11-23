#!/usr/bin/env python3

# Edited by Abdelrahman Younes @ University of Freiburg
# Email: younesa@cs.uni-freiburg.de

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.(https://github.com/facebookresearch/sound-spaces/blob/main/LICENSE)

import abc

import torch
import torch.nn as nn

from dynamical_simulator.dav_metrics import DSPL,DSNA
from ss_baselines.common.utils import CategoricalNetWithMask
from ss_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder
from ss_baselines.av_wan.models.visual_cnn import VisualCNN
from baselines.dav_nav.models.audio_autoencoder import AudioAutoEncoder
from baselines.dav_nav.models.audio_visual_cnn import AudioVisualCNN
from ss_baselines.av_wan.ppo.policy import Policy as BasePolicy
from ss_baselines.av_wan.ppo.policy import  Net
DUAL_GOAL_DELIMITER = ','

class Policy(BasePolicy):
    def __init__(self, net, dim_actions, masking=True):
        super().__init__(net,dim_actions,masking)

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

class AudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, encode_rgb, encode_depth, use_aux_loss):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._spectrogram = False

        self._use_aux_loss = use_aux_loss
        self.aux_loss = torch.nn.MSELoss()
        self._spectrogram = 'spectrogram' == self.goal_sensor_uuid

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
            audio_encoder_output, upsampling_output, aux = self.audio_auto_encoder(observations, aux=True)
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
