import numpy as np
import torch

from .obstacle import PolygonObstacle, CircleObstacle, LineObstacle
from utils import get_quad, PolygonGenerator, Agents


safe_padding = 4

class Stage(object):

    def __init__(self, env):
        self.env = env
        self.mask = torch.zeros([env.size[1], env.size[0]], device=env.device)

    def reset_stage(self):
        raise NotImplementedError

    def reset_agents(self) -> Agents:
        env = self.env
        env.crowd_map.map[0].zero_()

        # Generate positions without collision
        num_agents = env.num_agents
        all_pos = []
        for id in range(num_agents):
            # valid slots are those wihout obstacles, reward or agents
            # a mask is used to cancel out other custom slots
            valid_map = env.obs_map.map[0] + \
                env.obs_map.map[id + 1] + env.crowd_map.map[0] + self.mask
            valid_slots = torch.nonzero(valid_map == 0)

            idx = np.random.randint(valid_slots.shape[0])
            pos = valid_slots[idx].cpu().numpy()

            all_pos.append(pos)
            env.crowd_map.map[0, pos[1], pos[0]] = 1

        # Generate agents
        all_pos = np.array(all_pos)

        return Agents(
            all_pos.astype(float),
            np.zeros((num_agents, 2)),
            np.zeros((num_agents, 2))
        )
    
    # def _reset_mask(self):
    #     """
    #     Reset the agent mask and leave some space at the range
    #     """
    #     self.mask.zero_()
    #     for i in range(safe_padding):
    #         for j in range(self.env.size[1]):
    #             self.mask[j][i] = 1
    #             self.mask[j][self.env.size[0]-1-i] = 1
    #         for j in range(self.env.size[0]):
    #             self.mask[i][j] = 1
    #             self.mask[self.env.size[1]-1-i][j] = 1


class Stage_0(Stage):
    """
    Free ground with circle reward
    """

    def reset_stage(self):
        env = self.env
        env.obs_map.reset()

        # Stage bound
        points = get_quad(0, 0, env.size[0], env.size[1])
        bound = PolygonObstacle(points, reward = env.cfg.collide_penalty, fill=False)
        env.obs_map.add_polygon(bound)

        # Circle reward
        radius = 30
        while True:
            center = np.random.randint(safe_padding, env.size[0] - 1 - safe_padding, (2, ))
            if env.obs_map.map[0, center[0], center[1]] == 0:
                circle = CircleObstacle(center, radius, reward = env.cfg.reward_size)
                env.obs_map.add_circle(circle)
                break


class Stage_1(Stage):
    """
    One-way straight path way
    """

    def __init__(self, env):
        super(Stage_1, self).__init__(env)
        self.path_width = 30
        self.exit_size = 30
        self.obs_width = (env.size[0] - self.path_width) / 2

    def reset_stage(self):
        env = self.env
        env.obs_map.reset()

        # obstacle
        points = get_quad(0, 0, env.size[0], env.size[1])
        bound = PolygonObstacle(points, reward = env.cfg.collide_penalty)
        env.obs_map.add_polygon(bound)

        # pathway
        slope = np.random.uniform() * np.pi * 2
        line = LineObstacle(
            [env.size[1] // 2, env.size[0] // 2], slope, self.path_width, env.size[0] * 2, 0)
        env.obs_map.add_line(line)

        # reward
        self._add_reward(slope)

    def _add_reward(self, slope, reverse=False, agents=None):
        env = self.env
        side = -1 if reverse else 1
        
        if abs(np.sin(slope)) > np.sin(np.pi / 4):
            k = np.sin(slope)
        else:
            k = np.cos(slope)
        total = env.size[0] / abs(k)

        dx = self.exit_size * np.cos(slope) * side * 1.1
        dy = self.exit_size * np.sin(slope) * side * 1.1
        center = [env.size[1] / 2 + dy, env.size[0] / 2 + dx]
        line = LineObstacle(center, slope, self.path_width,
                            -(total - self.exit_size) / 2,
                            reward = env.cfg.reward_size)
        env.obs_map.add_line(line, agents)


class Stage_2(Stage_1):
    """
    Bidirectional straight path way
    """

    def reset_stage(self):
        env = self.env
        env.obs_map.reset()

        # obstacle
        points = get_quad(0, 0, env.size[0], env.size[1])
        bound = PolygonObstacle(points, reward = env.cfg.collide_penalty)
        env.obs_map.add_polygon(bound)

        # pathway
        slope = np.random.uniform() * np.pi * 2
        line = LineObstacle(
            [env.size[1] // 2, env.size[0] // 2], slope, self.path_width, env.size[0] * 2, 0)
        env.obs_map.add_line(line)

        # reward
        reverse = False
        for i in range(env.num_agents):
            self._add_reward(slope, reverse, [i])
            reverse = not reverse


class Stage_3(Stage_2):
    """
    Crossroads with bidirectional paths
    """

    def __init__(self, env):
        super(Stage_3, self).__init__(env)
        self.path_width = 30
        self.exit_size = 30
        self.obs_width = (env.size[0] - self.path_width) / 2

    def reset_stage(self):
        env = self.env
        env.obs_map.reset()

        # obstacle
        points = get_quad(0, 0, env.size[0], env.size[1])
        bound = PolygonObstacle(points, reward = env.cfg.collide_penalty)
        env.obs_map.add_polygon(bound)

        slope = np.random.uniform() * np.pi * 2
        # slope = np.pi / 6
        perpendicular = slope + np.pi / 2
        self._add_pathway(env, slope)
        self._add_pathway(env, perpendicular)

        # reward
        self._add_reward(slope, True, [i for i in range(0, env.num_agents, 4)])
        self._add_reward(slope, False, [i for i in range(1, env.num_agents, 4)])
        self._add_reward(perpendicular, True, [i for i in range(2, env.num_agents, 4)])
        self._add_reward(perpendicular, False, [i for i in range(3, env.num_agents, 4)])

    def _add_pathway(self, env, slope):
        # pathway
        line = LineObstacle(
            [env.size[1] // 2, env.size[0] // 2],
            slope, self.path_width, env.size[0] * 2, reward = 0)
        env.obs_map.add_line(line)


class Stage_4(Stage):
    """
    Bottleneck
    """

    def __init__(self, env):
        super(Stage_4, self).__init__(env)
        self.wall_thickness = 16
        self.door_width = 32
        self.exit_width = 32

    def reset_stage(self):
        env = self.env
        env.obs_map.reset()

        # agent spawning region
        self.exit_left = True if np.random.uniform() < 0.5 else False
        free_width = env.size[1] // 2
        self.mask.zero_()
        if self.exit_left:
            self.mask[:, :-free_width] = 1
        else:
            self.mask[:, free_width:] = 1

        # obstacle
        points = get_quad(0, 0, env.size[0], env.size[1])
        bound = PolygonObstacle(points, reward = env.cfg.collide_penalty)
        env.obs_map.add_polygon(bound)

        # pathway
        path = PolygonObstacle(
            get_quad(self.wall_thickness, 0, env.size[0] - self.wall_thickness * 2, env.size[1]),
            reward = 0)
        env.obs_map.add_polygon(path)

        # wall
        wall_pos = 0.25 if self.exit_left else 0.75
        wall_x = env.size[1] * wall_pos - self.wall_thickness // 2
        path = PolygonObstacle(
            get_quad(0, wall_x, env.size[0], self.wall_thickness),
            reward = env.cfg.collide_penalty)
        env.obs_map.add_polygon(path)

        # door
        # door_pos = np.random.rand() * 0.4 + 0.3
        door_pos = 0.5
        door_y = (env.size[0] - self.wall_thickness * 2) * door_pos - self.door_width // 2
        path = PolygonObstacle(
            get_quad(door_y, 0, self.door_width, env.size[1]),
            reward = 0)
        env.obs_map.add_polygon(path)

        # reward
        top_left = (self.wall_thickness, 0 if self.exit_left else env.size[1] - self.exit_width - 1)
        points = get_quad(*top_left, env.size[0] - self.wall_thickness * 2, self.exit_width)
        reward = PolygonObstacle(points, reward = env.cfg.reward_size)
        env.obs_map.add_polygon(reward)


def get_stage(env, stage_num):
    if stage_num == 0:
        return Stage_0(env)
    elif stage_num == 1:
        return Stage_1(env)
    elif stage_num == 2:
        return Stage_2(env)
    elif stage_num == 3:
        return Stage_3(env)
    elif stage_num == 4:
        return Stage_4(env)
    else:
        raise NotImplementedError
