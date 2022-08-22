# Edited by Abdelrahman Younes @ University of Freiburg
# Email: younesa@cs.uni-freiburg.de

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC
from typing import Any, List, Optional
# from collections import defaultdict, namedtuple
import logging
# import time
import pickle
import os
import torch
import torchaudio.transforms as T
# import torchaudio.functional as F

# import librosa
import scipy
from scipy.io import wavfile
from scipy.signal import fftconvolve
import numpy as np
import networkx as nx
from gym import spaces
import random

from habitat.core.registry import registry
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
# from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.simulator import (
    AgentState,
    Config,
    Observations,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
)
from soundspaces.utils import load_metadata


def overwrite_config(config_from: Config, config_to: Any) -> None:
    r"""Takes Habitat-API config and Habitat-Sim config structures. Overwrites
    Habitat-Sim config with Habitat-API values, where a field name is present
    in lowercase. Mostly used to avoid :ref:`sim_cfg.field = hapi_cfg.FIELD`
    code.
    Args:
        config_from: Habitat-API config node.
        config_to: Habitat-Sim config structure.
    """

    def if_config_to_lower(config):
        if isinstance(config, Config):
            return {key.lower(): val for key, val in config.items()}
        else:
            return config

    for attr, value in config_from.items():
        if hasattr(config_to, attr.lower()):
            setattr(config_to, attr.lower(), if_config_to_lower(value))


class DummySimulator:
    """
    Dummy simulator for avoiding loading the scene meshes when using cached observations.
    """

    def __init__(self):
        self.position = None
        self.rotation = None
        self._sim_obs = None
        self.source_position = None
        self.source_rotation = None

    def seed(self, seed):
        pass

    def set_agent_state(self, position, rotation):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = rotation

    def get_agent_state(self):
        class State:
            def __init__(self, position, rotation):
                self.position = position
                self.rotation = rotation

        return State(self.position, self.rotation)

    def set_source_state(self, source_position, source_rotation):
        self.source_position = np.array(source_position, dtype=np.float32)
        self.source_rotation = source_rotation

    def get_source_state(self):
        class State:
            def __init__(self, source_position, source_rotation):
                self.source_position = source_position
                self.source_rotation = source_rotation

        return State(self.source_position, self.source_rotation)

    def set_sensor_observations(self, sim_obs):
        self._sim_obs = sim_obs

    def get_sensor_observations(self):
        return self._sim_obs

    def close(self):
        pass


@registry.register_simulator()
class SoundSpacesSim(Simulator, ABC):
    r"""Changes made to simulator wrapper over habitat-sim
    This simulator first loads the graph of current environment and moves the agent among nodes.
    Any sounds can be specified in the episode and loaded in this simulator.
    Args:
        config: configuration for initializing the simulator.
    """

    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
        ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        self.config = config
        agent_config = self.get_agent_config()
        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene.id
        self._sim = habitat_sim.Simulator(self.sim_config)
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._prev_sim_obs = None

        self._source_position_index = None
        self._receiver_position_index = None
        self._rotation_angle = None
        self.source_rotation_angle = None
        self._current_sound = None
        self._source_sound_dict = dict()
        self._sampling_rate = None
        self._node2index = None
        self._frame_cache = dict()
        self._audiogoal_cache = dict()
        self._spectrogram_cache = dict()
        self._egomap_cache = dict()
        self._scene_observations = None
        self._episode_step_count = None
        self._is_episode_active = None
        self._position_to_index_mapping = dict()
        self._previous_step_collided = False

        self.points, self.graph = load_metadata(self.metadata_dir)
        for node in self.graph.nodes():
            self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node
        self._load_source_sounds()
        logging.info('Current scene: {} and sound: {}'.format(self.current_scene_name, self._current_sound))

        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.close()
            del self._sim
            self._sim = DummySimulator()
            with open(self.current_scene_observation_file, 'rb') as fo:
                self._frame_cache = pickle.load(fo)
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


    def create_sim_config(
            self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        overwrite_config(
            config_from=self.config.HABITAT_SIM_V0, config_to=sim_config
        )
        sim_config.scene.id = self.config.SCENE
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self.get_agent_config(), config_to=agent_config
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            sim_sensor_cfg = habitat_sim.SensorSpec()
            overwrite_config(
                config_from=sensor.config, config_to=sim_sensor_cfg
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )
            sim_sensor_cfg.parameters["hfov"] = str(sensor.config.HFOV)

            # accessing child attributes through parent interface
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type  # type: ignore
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.config.HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.config.ACTION_SPACE_CONFIG
        )(self.config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    def get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

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

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

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

    def get_observations_at(
            self,
            position: Optional[List[float]] = None,
            rotation: Optional[List[float]] = None,
            keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )
        if success:
            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    @property
    def binaural_rir_dir(self):
        return os.path.join(self.config.AUDIO.BINAURAL_RIR_DIR, self.config.SCENE_DATASET, self.current_scene_name)

    @property
    def source_sound_dir(self):
        return self.config.AUDIO.SOURCE_SOUND_DIR

    @property
    def metadata_dir(self):
        return os.path.join(self.config.AUDIO.METADATA_DIR, self.config.SCENE_DATASET, self.current_scene_name)

    @property
    def current_scene_name(self):
        # config.SCENE (_current_scene) looks like 'data/scene_datasets/replica/office_1/habitat/mesh_semantic.ply'
        return self._current_scene.split('/')[3]

    @property
    def current_scene_observation_file(self):
        return os.path.join(self.config.SCENE_OBSERVATION_DIR, self.config.SCENE_DATASET,
                            self.current_scene_name + '.pkl')

    @property
    def current_source_sound(self):
        return self._source_sound_dict[self._current_sound]

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
            self.source_rotation_angle = random.choice([0,90,180,270])
            if self.config['AUDIO']['val_test_dynamic']:
                self.current_source_type = "dynamic"
            else:
                self.current_source_type = random.choice(["static", "dynamic"])
            if self.current_source_type == "dynamic":
                self.motion_percentage = random.choice([10,20,30,40])
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

    @staticmethod
    def position_encoding(position):
        return '{:.2f}_{:.2f}_{:.2f}'.format(*position)

    def _position_to_index(self, position):
        if self.position_encoding(position) in self._position_to_index_mapping:
            return self._position_to_index_mapping[self.position_encoding(position)]
        else:
            raise ValueError("Position misalignment.")

    def _get_sim_observation(self):
        joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._frame_cache:
            return self._frame_cache[joint_index]
        else:
            assert not self.config.USE_RENDERED_OBSERVATIONS
            sim_obs = self._sim.get_sensor_observations()
            for sensor in sim_obs:
                sim_obs[sensor] = sim_obs[sensor]
            self._frame_cache[joint_index] = sim_obs
            return sim_obs

    def reset(self):
        logging.debug('Reset simulation')
        if not self.config.USE_RENDERED_OBSERVATIONS:
            sim_obs = self._sim.reset()
            if self._update_agents_state():
                sim_obs = self._get_sim_observation()
        else:
            sim_obs = self._get_sim_observation()
            self._sim.set_sensor_observations(sim_obs)

        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        self._previous_step_collided = False
        # Encapsulate data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

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
            addingMovingStep = random.choices([True, False], cum_weights=(self.motion_percentage, (100 - self.motion_percentage) ), k=1)
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

    def get_orientation(self):
        _base_orientation = 270
        return (_base_orientation - self._rotation_angle) % 360

    def get_source_orientation(self):
        _base_orientation = 270
        return (_base_orientation - self.source_rotation_angle) % 360

    @property
    def azimuth_angle(self):
        # this is the angle used to index the binaural audio files
        # in mesh coordinate systems, +Y forward, +X rightward, +Z upward
        # azimuth is calculated clockwise so +Y is 0 and +X is 90
        return -(self._rotation_angle + 0) % 360

    @property
    def reaching_goal(self):
        return self._source_position_index == self._receiver_position_index

    def _update_observations_with_audio(self, observations):
        audio = self.get_current_audio_observation()
        observations.update({"audio": audio})

    def _load_source_sounds(self):
        # load all mono files at once
        sound_files = os.listdir(self.source_sound_dir)
        for sound_file in sound_files:
            sr, audio_data = wavfile.read(os.path.join(self.source_sound_dir, sound_file))
            assert sr == 44100
            if sr != self.config.AUDIO.RIR_SAMPLING_RATE:
                audio_data = scipy.signal.resample(audio_data, self.config.AUDIO.RIR_SAMPLING_RATE)
            self._source_sound_dict[sound_file] = audio_data

    def _compute_euclidean_distance_between_sr_locations(self):
        p1 = self.graph.nodes[self._receiver_position_index]['point']
        p2 = self.graph.nodes[self._source_position_index]['point']
        d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2)
        return d

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

    def get_current_audiogoal_observation(self):
        sr_index = (self._source_position_index, self._receiver_position_index, self.azimuth_angle)
        if sr_index not in self._audiogoal_cache:
            self._audiogoal_cache[sr_index] = self._compute_audiogoal()

        return self._audiogoal_cache[sr_index]

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
            if self.moving_source and self.current_source_type == "dynamic" :
                position_b = self.get_source_state().source_position.tolist()
            index_b = self._position_to_index(position_b)
            assert index_a is not None and index_b is not None
            path_length = nx.shortest_path_length(self.graph, index_a, index_b) * self.config.GRID_SIZE
            distances.append(path_length)

        return min(distances)

    def get_straight_shortest_path_points(self, position_a, position_b):
        index_a = self._position_to_index(position_a)
        index_b = self._position_to_index(position_b)
        assert index_a is not None and index_b is not None

        shortest_path = nx.shortest_path(self.graph, source=index_a, target=index_b)
        points = list()
        for node in shortest_path:
            points.append(self.graph.nodes()[node]['point'])
        return points

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

    @property
    def previous_step_collided(self):
        return self._previous_step_collided

    def get_egomap_observation(self):
        joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._egomap_cache:
            return self._egomap_cache[joint_index]
        else:
            return None

    def cache_egomap_observation(self, egomap):
        self._egomap_cache[(self._receiver_position_index, self._rotation_angle)] = egomap

    def seed(self, seed):
        self._sim.seed(seed)
        random.seed(seed)
