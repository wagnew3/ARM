import numpy as np
import cv2
import json
import fileinput
import dm_control.mujoco as mujoco
from dm_control import suite

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import collections


class TouchTable(base.Task):

    def __init__(self):
        super(TouchTable, self).__init__()

    def initialize_episode(self, physics):
        super(TouchTable, self).initialize_episode(physics)
        

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        table_pos=physics.named.data.geom_xpos["tabletop"]
        wrist_pos=physics.named.data.geom_xpos["herb/wam_1//unnamed_geom_24"]
        return -np.sum(np.abs(table_pos-wrist_pos))


class IdiotThrowItForward(base.Task):
    def __init__(self, random=None):
        super(IdiotThrowItForward, self).__init__(random=random)

    def initialize_episode(self, physics):
        super(IdiotThrowItForward, self).initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        # return super().get_reward(physics)
        # return physics.named.model.body_pos['box_1'][1]
        # return physics.named.model.geom_pos['herb/wam_1/3_shoulder_pitch_link']
        # return 0
        return physics.named.data.geom_xpos['herb/wam_1//unnamed_geom_5'][1]

    # def get_termination(self, physics):
    #     # return super().get_termination(physics)
    #     return True


class EasyBlocks(base.Task):
    def __init__(self, random=None):
        super(EasyBlocks, self).__init__(random=random)

    def initialize_episode(self, physics):
        super(EasyBlocks, self).initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        return obs

    def get_distance(self, coordinate_1, coordinate_2):
        distance_matrix = coordinate_1 - coordinate_2
        return 1.0/max(1.0, (100*np.sum(np.abs(distance_matrix)))**2)

    def get_reward(self, physics):
        hand_pos=physics.named.data.geom_xpos["herb/wam_1//unnamed_geom_24"]
        block_pos=physics.named.data.geom_xpos["//unnamed_geom_8"]
        target_pos=physics.named.data.geom_xpos["//unnamed_geom_15"]
        
        robot_block_reward=-np.sum(np.abs(hand_pos-block_pos))
        block_target_reward=1-np.sum(np.abs(target_pos-block_pos))
        
        # return super().get_reward(physics)
        return (block_target_reward-0.568833897157993)

class HardBlocks(base.Task):
    def __init__(self, number_of_blocks, random=None):
        self.number_of_objects = number_of_blocks
        super(HardBlocks, self).__init__(random=random)

    def initialize_episode(self, physics):
        super(HardBlocks, self).initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        return obs

    def get_distance(self, coordinate_1, coordinate_2):
        distance_matrix = coordinate_1 - coordinate_2
        return -1 * np.linalg.norm(distance_matrix)

    def get_reward(self, physics):
        # return super().get_reward(physics)
        total_distance = 0
        for index in range(0, self.number_of_objects):
            total_distance += self.get_distance(
                physics.named.data.geom_xpos["box_to_move_" + str(index)], 
                physics.named.data.geom_xpos["target_to_reach_1"])
        return total_distance