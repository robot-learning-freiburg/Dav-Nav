# Edited by Abdelrahman Younes @ University of Freiburg
# Email: younesa@cs.uni-freiburg.de

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. (https://github.com/facebookresearch/sound-spaces/blob/main/LICENSE)
import logging
import os
import pickle
import random
from abc import ABC
from typing import Any, List, Optional

import habitat_sim
import networkx as nx
import numpy as np
import scipy
import torch
import torchaudio.transforms as T
from habitat.core.registry import registry
from habitat.core.simulator import (
    Config,
    SensorSuite,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
from scipy.io import wavfile
from scipy.signal import fftconvolve

from soundspaces.utils import load_metadata

from soundspaces.simulator import overwrite_config
from soundspaces.simulator import DummySimulator as BaseDummySimulator
from soundspaces.simulator import Simulator as BaseSimulator


class DummySimulator(BaseDummySimulator):
    """
    Dummy simulator for avoiding loading the scene meshes when using cached observations.
    """

    def __init__(self):
        super().__init__()
        self.source_position = None
        self.source_rotation = None

    def set_source_state(self, source_position, source_rotation):
        self.source_position = np.array(source_position, dtype=np.float32)
        self.source_rotation = source_rotation

    def get_source_state(self):
        class State:
            def __init__(self, source_position, source_rotation):
                self.source_position = source_position
                self.source_rotation = source_rotation

        return State(self.source_position, self.source_rotation)


@registry.register_simulator()
class SoundSpacesSim(BaseSimulator, ABC):
    r"""Changes made to simulator wrapper over habitat-sim
    This simulator first loads the graph of current environment and moves the agent among nodes.
    Any sounds can be specified in the episode and loaded in this simulator.
    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # self._sim = habitat_sim.Simulator(self.sim_config)

        self.source_rotation_angle = None
        self._egomap_cache = dict()

        if self.config['AUDIO']['moving_val_test_seed']:
            random.seed(self.config.SEED)
        self.moving_source = self.config['AUDIO']['moving_source']
        self.moving_source_goal_index = None
        self.moving_source_nodes_list = None
        self.moving_source_nodes_list_index = None
        self.source_position = None
        self.source_rotation = None
        self.augmentationType = "Noaugmentation"
        self.fMasking = None
        self.tMasking = None
        self.trainWithSecondAudio = self.config['AUDIO']['trainWithSecondAudio']
        self.trainWithSpecAugment = self.config['AUDIO']['trainWithSpecAugment']
        self.trainWithDistractor = self.config['AUDIO']['trainWithDistractor']
        self.trainWithRandmoizedSounds = False
        self.motion_percentage = 30
        if self.moving_source:
            self.current_source_type = None
            self.motion_percentage = 30
        if self.trainWithSecondAudio or self.trainWithDistractor:
            self.soundsForAugmentation = self._source_sound_dict.keys()
            valAndTestSounds = ["fan_6.wav", 'reverb_time.wav', 'fan_4.wav', 'terminal.wav', 'water_waves_2.wav',
                                'come_again.wav', 'infinitely.wav', 'person_8.wav', 'birds5.wav', 'person_7.wav',
                                'helicopter.wav', 'canon_short_2.wav', 'birds6.wav', 'arrived.wav', 'fan_7.wav',
                                'person_10.wav', 'radio_static.wav', 'telephone.wav', 'propeller.wav', 'person_11.wav',
                                'fan.wav', 'horn_2.wav', 'leak.wav', 'person_9.wav', 'creak.wav', 'beeps.wav',
                                'engine_4.wav', 'waves4.wav', 'turbine_4.wav']
            valSounds = ["fan_6.wav", 'reverb_time.wav', 'fan_4.wav', 'terminal.wav', 'water_waves_2.wav',
                         'come_again.wav', 'infinitely.wav', 'person_8.wav', 'birds5.wav', 'person_7.wav',
                         'helicopter.wav',
                         ]
            testSounds = ['canon_short_2.wav', 'arrived.wav', 'fan_7.wav',
                          'person_10.wav', 'radio_static.wav', 'telephone.wav', 'propeller.wav',
                          'person_11.wav',
                          'fan.wav', 'leak.wav', 'person_9.wav', 'creak.wav',
                          'engine_4.wav', 'waves4.wav', 'turbine_4.wav']

            if self.config['AUDIO']['val_sounds']:
                self.soundsForAugmentation = [x for x in self.soundsForAugmentation if x in valSounds]
            elif self.config['AUDIO']['test_sounds']:
                self.soundsForAugmentation = [x for x in self.soundsForAugmentation if x in testSounds]
            else:
                self.soundsForAugmentation = [x for x in self.soundsForAugmentation if x not in valAndTestSounds]
            if self.trainWithSecondAudio:
                self.secondAudio = None
                self.secondAudioName = None
                self.audioAugmentationType = "Noaugmentation"
            if self.trainWithDistractor:
                self.addingDistractor = None

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        if not self.config.USE_RENDERED_OBSERVATIONS:
            agent_state = self._sim.get_agent(agent_id).get_state()
        else:
            agent_state = self._sim.get_agent_state()

        return agent_state

    def get_source_state(self):
        if self.config.USE_RENDERED_OBSERVATIONS:
            source_state = self._sim.get_source_state()
        else:
            class State:
                def __init__(self, source_position, source_rotation):
                    self.source_position = source_position
                    self.source_rotation = source_rotation

            return State(self.source_position, self.source_rotation)

        return source_state

    def set_agent_state(
            self,
            position: List[float],
            rotation: List[float],
            agent_id: int = 0,
            reset_sensors: bool = True,
    ) -> bool:
        if not self.config.USE_RENDERED_OBSERVATIONS:
            agent = self._sim.get_agent(agent_id)
            new_state = self.get_agent_state(agent_id)
            new_state.position = position
            new_state.rotation = rotation
            new_state.sensor_states = {}
            agent.set_state(new_state, reset_sensors)
        else:
            pass

    def set_source_state(self, source_position, source_rotation):
        self.source_position = np.array(source_position, dtype=np.float32)
        self.source_rotation = source_rotation

    def reconfigure(self, config: Config) -> None:
        self.config = config
        is_same_sound = config.AGENT_0.SOUND_ID == self._current_sound
        if not is_same_sound:
            self._current_sound = self.config.AGENT_0.SOUND_ID
        if self.trainWithRandmoizedSounds:
            self._current_sound = random.choice(self.soundsForAugmentation)
        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            logging.debug('Current scene: {} and sound: {}'.format(self.current_scene_name, self._current_sound))

            if not self.config.USE_RENDERED_OBSERVATIONS:
                self._sim.close()
                del self._sim
                self.sim_config = self.create_sim_config(self._sensor_suite)
                self._sim = habitat_sim.Simulator(self.sim_config)
                self._update_agents_state()
                self._frame_cache = dict()
            else:
                with open(self.current_scene_observation_file, 'rb') as fo:
                    self._frame_cache = pickle.load(fo)
            logging.debug('Loaded scene {}'.format(self.current_scene_name))

            self.points, self.graph = load_metadata(self.metadata_dir)
            for node in self.graph.nodes():
                self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node

        if not self.trainWithSecondAudio and not self.trainWithDistractor:
            if not is_same_scene or not is_same_sound:
                self._audiogoal_cache = dict()
                self._spectrogram_cache = dict()

            if not is_same_scene:
                self._egomap_cache = dict()
        else:
            self._audiogoal_cache = dict()
            self._spectrogram_cache = dict()
            self._egomap_cache = dict()

        self._episode_step_count = 0

        # set agent positions
        self._receiver_position_index = self._position_to_index(self.config.AGENT_0.START_POSITION)
        self._source_position_index = self._position_to_index(self.config.AGENT_0.GOAL_POSITION)
        # the agent rotates about +Y starting from -Z counterclockwise,
        # so rotation angle 90 means the agent rotate about +Y 90 degrees
        self._rotation_angle = int(np.around(np.rad2deg(quat_to_angle_axis(quat_from_coeffs(
            self.config.AGENT_0.START_ROTATION))[0]))) % 360
        if self.moving_source:
            self.source_rotation_angle = random.choice([0, 90, 180, 270])
            if self.config['AUDIO']['val_test_dynamic']:
                self.current_source_type = "dynamic"
            else:
                self.current_source_type = random.choice(["static", "dynamic"])
            if self.current_source_type == "dynamic":
                self.motion_percentage = random.choice([10, 20, 30, 40])
                if not self.config.USE_RENDERED_OBSERVATIONS:
                    self.set_source_state(list(self.graph.nodes[self._source_position_index]['point']),
                                          quat_from_angle_axis(np.deg2rad(self.source_rotation_angle),
                                                               np.array([0, 1, 0])))
                else:
                    self._sim.set_source_state(list(self.graph.nodes[self._source_position_index]['point']),
                                               quat_from_angle_axis(np.deg2rad(self.source_rotation_angle),
                                                                    np.array([0, 1, 0])))

                self.set_new_goal()
                while self.get_straight_shortest_path_nodes(self._source_position_index,
                                                            self.moving_source_goal_index) is None:
                    self.set_new_goal()
                self.moving_source_nodes_list = self.get_straight_shortest_path_nodes(self._source_position_index,
                                                                                      self.moving_source_goal_index)
                self.moving_source_nodes_list_index = 1

        if not self.config.USE_RENDERED_OBSERVATIONS:
            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 self.config.AGENT_0.START_ROTATION)
        else:
            self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                      quat_from_coeffs(self.config.AGENT_0.START_ROTATION))

        logging.debug("Initial source, agent at: {}, {}, orientation: {}".
                      format(self._source_position_index, self._receiver_position_index, self.get_orientation()))

        if self.trainWithSecondAudio:
            self.audioAugmentationType = random.choice(["Noaugmentation", "SecondAudio"])
            if self.audioAugmentationType == "SecondAudio":
                self.secondAudioName = random.choice(self.soundsForAugmentation)
                self.secondAudio = self._source_sound_dict[self.secondAudioName]
        if self.trainWithDistractor:
            self.addingDistractor = bool(random.getrandbits(1))

    def step(self, action, only_allowed=True):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.
        :param action: action to be taken
        :param only_allowed: if true, then can't step anywhere except allowed locations
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )

        self._previous_step_collided = False
        # STOP: 0, FORWARD: 1, LEFT: 2, RIGHT: 2
        if action == HabitatSimActions.STOP:
            self._is_episode_active = False
        else:
            prev_position_index = self._receiver_position_index
            prev_rotation_angle = self._rotation_angle
            if action == HabitatSimActions.MOVE_FORWARD:
                # the agent initially faces -Z by default
                self._previous_step_collided = True
                for neighbor in self.graph[self._receiver_position_index]:
                    p1 = self.graph.nodes[self._receiver_position_index]['point']
                    p2 = self.graph.nodes[neighbor]['point']
                    direction = int(np.around(np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
                    if direction == self.get_orientation():
                        self._receiver_position_index = neighbor
                        self._previous_step_collided = False
                        break
            elif action == HabitatSimActions.TURN_LEFT:
                # agent rotates counterclockwise, so turning left means increasing rotation angle by 90
                self._rotation_angle = (self._rotation_angle + 90) % 360
            elif action == HabitatSimActions.TURN_RIGHT:
                self._rotation_angle = (self._rotation_angle - 90) % 360

            if self.config.CONTINUOUS_VIEW_CHANGE:
                intermediate_observations = list()
                fps = self.config.VIEW_CHANGE_FPS
                if action == HabitatSimActions.MOVE_FORWARD:
                    prev_position = np.array(self.graph.nodes[prev_position_index]['point'])
                    current_position = np.array(self.graph.nodes[self._receiver_position_index]['point'])
                    for i in range(1, fps):
                        intermediate_position = prev_position + i / fps * (current_position - prev_position)
                        self.set_agent_state(intermediate_position.tolist(), quat_from_angle_axis(np.deg2rad(
                            self._rotation_angle), np.array([0, 1, 0])))
                        sim_obs = self._sim.get_sensor_observations()
                        observations = self._sensor_suite.get_observations(sim_obs)
                        intermediate_observations.append(observations)
                else:
                    for i in range(1, fps):
                        if action == HabitatSimActions.TURN_LEFT:
                            intermediate_rotation = prev_rotation_angle + i / fps * 90
                        elif action == HabitatSimActions.TURN_RIGHT:
                            intermediate_rotation = prev_rotation_angle - i / fps * 90
                        self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                             quat_from_angle_axis(np.deg2rad(intermediate_rotation),
                                                                  np.array([0, 1, 0])))
                        sim_obs = self._sim.get_sensor_observations()
                        observations = self._sensor_suite.get_observations(sim_obs)
                        intermediate_observations.append(observations)

            if not self.config.USE_RENDERED_OBSERVATIONS:
                self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                     quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
            else:
                self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                          quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
            addingMovingStep = random.choices([True, False],
                                              cum_weights=(self.motion_percentage, (100 - self.motion_percentage)), k=1)
            if self.moving_source and addingMovingStep[0] and self.current_source_type == "dynamic":
                if self.moving_source_nodes_list_index < len(self.moving_source_nodes_list):
                    while not os.path.exists(
                            os.path.join(self.binaural_rir_dir, str(self.azimuth_angle), '{}_{}.wav'.format(
                                self._receiver_position_index,
                                self.moving_source_nodes_list[self.moving_source_nodes_list_index]))):
                        self.moving_source_nodes_list_index += 1
                else:
                    self.set_new_goal()
                    while self.get_straight_shortest_path_nodes(self._source_position_index,
                                                                self.moving_source_goal_index) is None:
                        self.set_new_goal()
                    self.moving_source_nodes_list = self.get_straight_shortest_path_nodes(
                        self._source_position_index,
                        self.moving_source_goal_index)
                    self.moving_source_nodes_list_index = 1

                self._source_position_index = self.moving_source_nodes_list[self.moving_source_nodes_list_index]
                self.moving_source_nodes_list_index += 1

            if self.moving_source:
                if not self.config.USE_RENDERED_OBSERVATIONS:
                    self.set_source_state(list(self.graph.nodes[self._source_position_index]['point']),
                                          quat_from_angle_axis(np.deg2rad(self.source_rotation_angle),
                                                               np.array([0, 1, 0])))
                else:
                    self._sim.set_source_state(list(self.graph.nodes[self._source_position_index]['point']),
                                               quat_from_angle_axis(np.deg2rad(self.source_rotation_angle),
                                                                    np.array([0, 1, 0])))

        self._episode_step_count += 1

        # log debugging info
        logging.debug('After taking action {}, s,r: {}, {}, orientation: {}, location: {}'.format(
            action, self._source_position_index, self._receiver_position_index,
            self.get_orientation(), self.graph.nodes[self._receiver_position_index]['point']))

        sim_obs = self._get_sim_observation()
        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_sensor_observations(sim_obs)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        if self.config.CONTINUOUS_VIEW_CHANGE:
            observations['intermediate'] = intermediate_observations

        return observations

    def _compute_audiogoal(self):
        binaural_rir_file = os.path.join(self.binaural_rir_dir, str(self.azimuth_angle), '{}_{}.wav'.format(
            self._receiver_position_index, self._source_position_index))
        try:
            sampling_freq, binaural_rir = wavfile.read(binaural_rir_file)  # float32
            # # pad RIR with zeros to take initial delays into account
            # num_delay_sample = int(self._compute_euclidean_distance_between_sr_locations() / 343.0 * sampling_freq)
            # binaural_rir = np.pad(binaural_rir, ((num_delay_sample, 0), (0, 0)))

        except ValueError:
            logging.warning("{} file is not readable".format(binaural_rir_file))
            binaural_rir = np.zeros((self.config.AUDIO.RIR_SAMPLING_RATE, 2)).astype(np.float32)
        if len(binaural_rir) == 0:
            logging.debug("Empty RIR file at {}".format(binaural_rir_file))
            binaural_rir = np.zeros((self.config.AUDIO.RIR_SAMPLING_RATE, 2)).astype(np.float32)

        # by default, convolve in full mode, which preserves the direct sound
        binaural_convolved = [fftconvolve(self.current_source_sound, binaural_rir[:, channel]
                                          ) for channel in range(binaural_rir.shape[-1])]
        audiogoal = np.array(binaural_convolved)[:, :self.current_source_sound.shape[0]]

        if self.trainWithSecondAudio and self.audioAugmentationType == "SecondAudio":
            # convolve second sound with RIR
            binaural_second_audio_convolved = [
                fftconvolve(self.secondAudio, binaural_rir[:, channel]
                            ) for channel in range(binaural_rir.shape[-1])]
            audiogoal += np.array(binaural_second_audio_convolved)[:, :self.current_source_sound.shape[0]]

        addingDistractorStep = bool(random.getrandbits(1))
        if self.trainWithDistractor and self.addingDistractor and addingDistractorStep:
            distractor_position_index = random.choice(list(self._position_to_index_mapping.values()))
            while not os.path.exists(os.path.join(self.binaural_rir_dir, str(self.azimuth_angle), '{}_{}.wav'.format(
                    self._receiver_position_index,
                    distractor_position_index))) or distractor_position_index == self._receiver_position_index or distractor_position_index == self._source_position_index:
                distractor_position_index = random.choice(list(self._position_to_index_mapping.values()))

            distractor_binaural_rir_file = os.path.join(self.binaural_rir_dir, str(self.azimuth_angle),
                                                        '{}_{}.wav'.format(
                                                            self._receiver_position_index, distractor_position_index))
            try:
                sampling_freq, distractor_binaural_rir = wavfile.read(distractor_binaural_rir_file)  # float32

            except ValueError:
                logging.warning("{} file is not readable".format(distractor_binaural_rir_file))
                distractor_binaural_rir = np.zeros((self.config.AUDIO.RIR_SAMPLING_RATE, 2)).astype(np.float32)
            if len(distractor_binaural_rir) == 0:
                logging.debug("Empty RIR file at {}".format(distractor_binaural_rir_file))
                distractor_binaural_rir = np.zeros((self.config.AUDIO.RIR_SAMPLING_RATE, 2)).astype(np.float32)

            distractor_sound_name = random.choice(self.soundsForAugmentation)
            while distractor_sound_name == self._current_sound or distractor_sound_name == self.secondAudioName:
                distractor_sound_name = random.choice(self.soundsForAugmentation)
            distractor_sound = self._source_sound_dict[distractor_sound_name]
            # convolve distractor sound
            distractor_binaural_convolved = [fftconvolve(distractor_sound, distractor_binaural_rir[:, channel]
                                                         ) for channel in range(distractor_binaural_rir.shape[-1])]
            audiogoal += np.array(distractor_binaural_convolved)[:, :self.current_source_sound.shape[0]]

        return audiogoal

    def get_current_spectrogram_observation(self, audiogoal2spectrogram):
        sr_index = (self._source_position_index, self._receiver_position_index)
        sr_index = sr_index + (self.azimuth_angle,)
        if sr_index not in self._spectrogram_cache:
            audiogoal = self._compute_audiogoal()
            spectrogram = audiogoal2spectrogram(audiogoal)
            if not self.trainWithSpecAugment:
                self._spectrogram_cache[sr_index] = spectrogram
            else:
                augmentationType = random.choice(["Noaugmentation", "FrequencyMasking", "TimeMasking", "both"])
                if augmentationType == "Noaugmentation":
                    self._spectrogram_cache[sr_index] = spectrogram
                else:
                    spectorgramTensor = torch.from_numpy(spectrogram).permute(2, 0, 1)
                    if augmentationType == "FrequencyMasking":
                        masking = T.FrequencyMasking(freq_mask_param=12)
                        spec = masking(spectorgramTensor, spectorgramTensor.mean())
                        self._spectrogram_cache[sr_index] = spec.permute(1, 2, 0).numpy()
                    elif augmentationType == "TimeMasking":
                        masking = T.TimeMasking(time_mask_param=self.config['AUDIO']['time_mask_param'])
                        spec = masking(spectorgramTensor, spectorgramTensor.mean())
                        self._spectrogram_cache[sr_index] = spec.permute(1, 2, 0).numpy()
                    else:
                        fMasking = T.FrequencyMasking(freq_mask_param=12)
                        spec = fMasking(spectorgramTensor, spectorgramTensor.mean())
                        tMasking = T.TimeMasking(time_mask_param=self.config['AUDIO']['time_mask_param'])
                        spec = tMasking(spec, spectorgramTensor.mean())
                        self._spectrogram_cache[sr_index] = spec.permute(1, 2, 0).numpy()

        return self._spectrogram_cache[sr_index]

    def geodesic_distance(self, position_a, position_bs, episode=None):
        distances = []
        for position_b in position_bs:
            index_a = self._position_to_index(position_a)
            # update position b if moving source
            if self.moving_source and self.current_source_type == "dynamic":
                position_b = self.get_source_state().source_position.tolist()
            index_b = self._position_to_index(position_b)
            assert index_a is not None and index_b is not None
            path_length = nx.shortest_path_length(self.graph, index_a, index_b) * self.config.GRID_SIZE
            distances.append(path_length)

        return min(distances)

    def get_straight_shortest_path_nodes(self, index_a, index_b):
        assert index_a is not None and index_b is not None
        try:
            shortest_path = nx.shortest_path(self.graph, source=index_a, target=index_b)
            return shortest_path
        except nx.NetworkXNoPath:
            return None

    def set_new_goal(self):
        self.moving_source_goal_index = random.choice(list(self._position_to_index_mapping.values()))
        if self.config['AUDIO']['val_test_dynamic']:
            while not os.path.exists(os.path.join(self.binaural_rir_dir, str(self.azimuth_angle), '{}_{}.wav'.format(
                    self._receiver_position_index,
                    self.moving_source_goal_index))) or self.moving_source_goal_index == self._source_position_index:
                self.moving_source_goal_index = random.choice(list(self._position_to_index_mapping.values()))
        else:
            while not os.path.exists(os.path.join(self.binaural_rir_dir, str(self.azimuth_angle), '{}_{}.wav'.format(
                    self._receiver_position_index,
                    self.moving_source_goal_index))) or self.moving_source_goal_index == self._receiver_position_index or self.moving_source_goal_index == self._source_position_index:
                self.moving_source_goal_index = random.choice(list(self._position_to_index_mapping.values()))

    def get_egomap_observation(self):
        joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._egomap_cache:
            return self._egomap_cache[joint_index]
        else:
            return None

    def seed(self, seed):
        self._sim.seed(seed)
        random.seed(seed)
