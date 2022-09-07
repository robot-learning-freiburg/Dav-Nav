# Edited by Abdelrahman Younes @ University of Freiburg
# Email: younesa@cs.uni-freiburg.de

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. (https://github.com/facebookresearch/sound-spaces/blob/main/LICENSE)

from typing import Any, Union
import networkx as nx
import numpy as np
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import (
    Simulator,
)
from habitat.tasks.nav.nav import Measure, EmbodiedTask, Success
from habitat_sim.utils.common import quat_from_coeffs, quat_to_angle_axis

@registry.register_measure
class DSPL(Measure):
    r"""DSPL (Dynamic Success weighted by Path Length)
        """

    def __init__(
            self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "dspl"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._agent_start_index = self._sim._position_to_index(episode.start_position)
        self._agent_start_rotation = int(
            np.around(np.rad2deg(quat_to_angle_axis(quat_from_coeffs(episode.start_rotation))[0]))) % 360
        self._found_closest_goal = False
        self._metric = None
        self._geodesic_distance = episode.info['geodesic_distance']
        self._shortest_intersection_point = None

    def get_orientation(self):
        _base_orientation = 270
        return (_base_orientation - self._rotation_angle) % 360

    def update_metric(
            self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        if not self._found_closest_goal:
            self._rotation_angle = self._agent_start_rotation
            self._source_position_index = self._sim._source_position_index
            self._episode_step_count = self._sim._episode_step_count
            self._shortest_path_nodes = self._sim.get_straight_shortest_path_nodes(self._agent_start_index,
                                                                                   self._source_position_index)
            self._required_action_count = 0
            points = list()
            for node in self._shortest_path_nodes[1:]:
                points.append(self._sim.graph.nodes()[node]['point'])

            p1 = self._sim.graph.nodes()[self._agent_start_index]['point']
            for p2 in points:
                direction = int(np.around(np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
                if direction == self.get_orientation():
                    p1 = p2
                elif direction < self.get_orientation():
                    self._rotation_angle = (self._rotation_angle + 90) % 360
                elif direction > self.get_orientation():
                    self._rotation_angle = (self._rotation_angle - 90) % 360
                self._required_action_count += 1
            if self._required_action_count <= self._episode_step_count:
                self._found_closest_goal = True
                self._geodesic_distance = nx.shortest_path_length(self._sim.graph, self._agent_start_index,
                                                                  self._source_position_index) * self._sim.config.GRID_SIZE
                self._shortest_intersection_point = self._sim.graph.nodes()[self._source_position_index]['point']
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        self._metric = ep_success * (
                self._geodesic_distance / max(task.measurements.measures['spl']._agent_episode_distance,
                                              self._geodesic_distance))

@registry.register_measure
class DSNA(Measure):
    r"""DSPL (Dynamic Success weighted by Path Length)
    """

    def __init__(
            self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._start_end_num_action = None
        self._agent_num_action = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "dsna"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._start_end_num_action = episode.info["num_action"]
        self._agent_num_action = 0
        self._metric = None

    def update_metric(
            self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        self._agent_num_action += 1
        # here will calculate it for the dynamic task and reported in the paper as (DSNA)
        self._metric = ep_success * (
                task.measurements.measures['dspl']._required_action_count
                / max(
            task.measurements.measures['dspl']._required_action_count, self._agent_num_action
        )
        )

