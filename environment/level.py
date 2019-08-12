import numpy as np
import torch

from .obstacle import PolygonObstacle, CircleObstacle, LineObstacle
from utils import get_quad, PolygonGenerator, Agents


class Level(object):

    def __init__(self, env):
        self.env = env
        self.mask = torch.zeros([env.size[1], env.size[0]], device=env.device)

    def reset_level(self):
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


class Level_0(Level):
    """
    Free ground with circle reward
    """

    def reset_level(self):
        env = self.env
        env.obs_map.reset()

        # Stage bound
        points = get_quad(0, 0, env.size[0] - 1, env.size[1] - 1)
        bound = PolygonObstacle(points, reward=-10, fill=False)
        env.obs_map.add_polygon(bound)

        # Circle reward
        # radius = np.random.randint(10, 30)
        radius = 30
        while True:
            center = np.random.randint(0, env.size[0] - 1, (2, ))
            if env.obs_map.map[0, center[0], center[1]] == 0:
                circle = CircleObstacle(center, radius, reward=10)
                env.obs_map.add_circle(circle)
                break


class Level_1(Level):
    """
    One-way straight path way
    """

    def __init__(self, env):
        super(Level_1, self).__init__(env)
        self.path_width = 30
        self.exit_size = 30
        self.obs_width = (env.size[0] - self.path_width) / 2

    def reset_level(self):
        env = self.env
        env.obs_map.reset()

        # obstacle
        points = get_quad(0, 0, env.size[0] - 1, env.size[1] - 1)
        bound = PolygonObstacle(points, reward=-10)
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
        total = env.size[0] / abs(np.sin(slope) if abs(np.sin(slope))
                                  > np.sin(np.pi / 4) else np.cos(slope))
        dx = self.exit_size * np.cos(slope) * side
        dy = self.exit_size * np.sin(slope) * side
        center = [env.size[1] // 2 + dy, env.size[0] // 2 + dx]
        line = LineObstacle(center, slope, self.path_width,
                            -(total - self.exit_size) / 2, 10)
        env.obs_map.add_line(line, agents)


class Level_2(Level_1):
    """
    Bidirectional straight path way
    """

    def reset_level(self):
        env = self.env
        env.obs_map.reset()

        # obstacle
        points = get_quad(0, 0, env.size[0] - 1, env.size[1] - 1)
        bound = PolygonObstacle(points, reward=-10)
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


class Level_3(Level):
    """
    Bottleneck
    """

    def __init__(self, env):
        super(Level_3, self).__init__(env)
        self.wall_thickness = 32
        self.neck_width = 8
        self.exit_width = 32

    def reset_level(self):
        env = self.env
        env.obs_map.reset()

        self.exit_left = True if np.random.uniform() < 0.5 else False
        free_width = env.size[1] // 4
        self.mask.zero_()
        if self.exit_left:
            self.mask[:, :-free_width] = 1
        else:
            self.mask[:, free_width:] = 1

        # obstacle
        points = get_quad(0, 0, env.size[0] - 1, env.size[1] - 1)
        bound = PolygonObstacle(points, reward=-10)
        env.obs_map.add_polygon(bound)

        # pathway
        x_length = (env.size[1] - self.wall_thickness) // 2 - 1
        y_length = env.size[0] - self.wall_thickness * 2 - 1
        
        path = PolygonObstacle(
            get_quad(self.wall_thickness, 0, y_length, x_length), reward=0)
        env.obs_map.add_polygon(path)

        path = PolygonObstacle(
            get_quad(self.wall_thickness, x_length + self.wall_thickness, y_length, x_length), reward=0)
        env.obs_map.add_polygon(path)

        path = PolygonObstacle(
            get_quad((env.size[0] - self.neck_width) // 2, x_length, self.neck_width, self.wall_thickness), reward=0)
        env.obs_map.add_polygon(path)

        # reward
        top_left = (self.wall_thickness, 0 if self.exit_left else env.size[1] - self.exit_width - 1)
        points = get_quad(*top_left, y_length, self.exit_width)
        reward = PolygonObstacle(points, reward = 10)
        env.obs_map.add_polygon(reward)


class Level_4(Level):
    """
    Crossroads with bidirectional paths
    """

    def __init__(self, env):
        super(Level_4, self).__init__(env)
        self.path_width = 30
        self.exit_size = 30
        self.obs_width = (env.size[0] - self.path_width) / 2

    def reset_level(self):
        env = self.env
        env.obs_map.reset()

        # obstacle
        points = get_quad(0, 0, env.size[0] - 1, env.size[1] - 1)
        bound = PolygonObstacle(points, reward=-10)
        env.obs_map.add_polygon(bound)

        slope = np.random.uniform() * np.pi * 2
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
            [env.size[1] // 2, env.size[0] // 2], slope, self.path_width, env.size[0] * 2, 0)
        env.obs_map.add_line(line)

    def _add_reward(self, slope, reverse=False, agents=None):
        env = self.env
        side = -1 if reverse else 1
        total = env.size[0] / abs(np.sin(slope) if abs(np.sin(slope))
                                  > np.sin(np.pi / 4) else np.cos(slope))
        dx = self.exit_size * np.cos(slope) * side
        dy = self.exit_size * np.sin(slope) * side
        center = [env.size[1] // 2 + dy, env.size[0] // 2 + dx]
        line = LineObstacle(center, slope, self.path_width,
                            -(total - self.exit_size) / 2, 10)
        env.obs_map.add_line(line, agents)


def get_level(env, level_num):
    if level_num == 0:
        return Level_0(env)
    elif level_num == 1:
        return Level_1(env)
    elif level_num == 2:
        return Level_2(env)
    elif level_num == 3:
        return Level_3(env)
    elif level_num == 4:
        return Level_4(env)
    else:
        raise NotImplementedError
