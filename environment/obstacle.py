import numpy as np
import torch
import itertools

from utils import to_pixel
from .feature_map import FeatureMap

EPS = 1e-6


class PolygonObstacle():
    """
    A polygon obstacle
    """

    def __init__(self, points, reward=-5, auto_close=True, fill=True):
        """
        Please make sure the lines don't intersect with each other

        Parameters:
            points          A numpy array with shape (n, 2). Point coords must be [y, x]
            auto_close      Wether to connect the last point with the first point
            fill            Wether the obstacle is solid
        """
        if auto_close:
            self.points = np.vstack(
                (points, points[np.newaxis, 0])).astype(int)
        else:
            self.points = points.astype(int)
        self.min = np.min(points, axis=0)
        self.max = np.max(points, axis=0)
        self.fill = fill
        self.reward = reward


class CircleObstacle():
    """
    A circle obstacle
    """

    def __init__(self, center, radius, reward=-5, fill=True):
        self.center = center
        self.radius = radius
        self.reward = reward
        self.fill = fill


class LineObstacle():
    """
    A line obstacle with width
    """

    def __init__(self, point, slope, width, half_length, reward, fill=True):
        """
        Construct a line with a set point and a slope

        params:
        point:              np array of size (2, )
        slope:              slope in radius
        width:              line width
        half_length:        for how long can the line stretch from the given point, negative for reverted lines
        """
        if abs(np.sin(slope)) < 0.99:
            self.vertical = False
            self.A = np.tan(slope)
            self.B = -1
            self.C = point[0] - self.A * point[1]
            self.sqrt = np.sqrt(self.A ** 2 + self.B ** 2)
        else:
            self.vertical = True

        self.x = point[1]
        self.y = point[0]

        self.width = width
        self.reward = reward
        self.fill = fill
        self.length = half_length


class ObstacleMap(FeatureMap):
    def __init__(self, device, h, w, num_agents):
        """
        A two-channel feature map

        map[0]:     obstacle map
        map[1:]:    reward maps
        """
        super(ObstacleMap, self).__init__(device, h, w, num_agents + 1)
        self.obs_color = torch.tensor(
            [1, 0, 0], device=device, dtype=torch.float).view(3, 1)
        self.reward_color = torch.tensor(
            [1, 1, 0], device=device, dtype=torch.float).view(3, 1)

        self.num_agents = num_agents

    def render(self, img):
        """
        Add feature map contents to an image buffer

        parameters:
            img:    Tensor with shape (3, h, w)
        """
        idx = torch.nonzero(self.map[1:].sum(dim=0))
        img[:, idx[:, 0], idx[:, 1]] = self.reward_color
        idx = torch.nonzero(self.map[0])
        img[:, idx[:, 0], idx[:, 1]] = self.obs_color

    def get_feature(self):
        """
        Return the feature map, size (N, 2, H, W)
        """
        normalized = self._normalized()
        obstacles = normalized[0].unsqueeze(0).expand(
            self.num_agents, -1, -1, -1)  # (H, W) to (N, 1, H, W)
        rewards = normalized[1:].unsqueeze(1)   # (N, H, W) to (N, 1, H, W)
        return torch.cat((obstacles, rewards), dim=1)      # (N, 2, H, W)

    def add_polygon(self, obs: PolygonObstacle, agents=None):
        """
        Polygon obstacle or reward

        Zero-reward can clear previous reward

        params:
            agents:         the agents' id that this obstacle applies to, must be array or tuple. None for all agents
        """
        buffer = np.zeros(self.size, dtype=np.float32)
        valid_reward = abs(obs.reward) if obs.reward != 0 else 1

        for p1, p2 in zip(obs.points[:-1], obs.points[1:]):
            self._add_line_segment(buffer, p1, p2, valid_reward)
        if obs.fill:
            self._fill_area(buffer, valid_reward)

        targets = self._get_target(obs.reward, agents)
        idx = np.nonzero(buffer)
        for target in targets:
            if obs.reward != 0:
                self.map[target, idx[0], idx[1]] = torch.from_numpy(
                    buffer[idx[0], idx[1]]).to(device=self.device)
            else:
                self.map[target, idx[0], idx[1]] = 0

    def add_line(self, obs: LineObstacle, agents=None):
        """
        Add line obstacle or reward

        Zero-reward can clear previous reward

        params:
            agents:         the agents' id that this obstacle applies to, must be array or tuple. None for all agents
        """
        if obs.fill:
            idx = np.array(list(itertools.product(
                range(self.size[0]), range(self.size[1]))))
            if obs.vertical:
                w_mask = abs(idx[:, 0] - obs.x) <= obs.width
                if obs.length > 0:
                    l_mask = np.abs(idx[:, 1] - obs.y) <= obs.length
                else:
                    l_mask = np.abs(idx[:, 1] - obs.y) > -obs.length
            else:
                y_dist = np.abs(obs.A * idx[:, 0] + obs.B * idx[:, 1] +
                                obs.C) / obs.sqrt
                w_mask = y_dist <= obs.width
                x_dist = np.sqrt(np.maximum(
                    (idx[:, 1] - obs.y) ** 2 + (idx[:, 0] - obs.x) ** 2 - y_dist ** 2, 0))
                if obs.length > 0:
                    l_mask = x_dist <= obs.length
                else:
                    l_mask = x_dist > -obs.length

            mask = np.logical_and(w_mask, l_mask)
            targets = self._get_target(obs.reward, agents)
            for target in targets:
                self.map[target, idx[mask, 0],
                         idx[mask, 1]] = abs(obs.reward)
        else:
            raise NotImplementedError

    def add_circle(self, obs: CircleObstacle, agents=None):
        """
        Add circle obstacle or reward

        params:
            agents:         the agents' id that this obstacle applies to, must be array or tuple. None for all agents
        """
        if obs.fill:
            idx = np.array(list(itertools.product(
                range(self.size[0]), range(self.size[1]))))
            mask = np.square(idx - np.array(obs.center)
                             [None, :]).sum(axis=1) <= obs.radius ** 2
            targets = self._get_target(obs.reward, agents)
            for target in targets:
                self.map[target, idx[mask, 0],
                        idx[mask, 1]] = abs(obs.reward)
        else:
            raise NotImplementedError

    def _add_line_segment(self, buffer, p1, p2, reward):
        """
        Add line p1-p2 to the obstacle map

        Use a scan-line algorithm to visit a line of pixels
        """
        # make sure p2 is on the right side
        if p1[1] > p2[1]:
            self._add_line_segment(buffer, p2, p1, reward)
            return

        k = (p2[0] - p1[0]) / (p2[1] - p1[1] + EPS)
        if abs(k) > 1:
            unit = 1 if k > 0 else -1
            delta = np.array([unit, unit / k])
            total_step = (int(p2[0]) - int(p1[0])) * unit + 1
        else:
            delta = np.array([k, 1])
            total_step = int(p2[1]) - int(p1[1]) + 1

        p = p1.astype(float)
        pxls = np.zeros((total_step, 2), dtype=int)
        for i in range(total_step):
            pxls[i] = to_pixel(p)
            p += delta
        buffer[pxls[:, 0], pxls[:, 1]] = reward

    def _fill_area(self, buffer, reward):
        """
        Fill area between drawn lines using scan-line method
        """
        for row in range(self.size[0]):
            filling = False
            last_empty = True
            for col in range(self.size[1]):
                this_empty = buffer[row, col] == 0
                if filling:
                    if this_empty:
                        buffer[row, col] = reward
                    else:
                        filling = False
                elif buffer[row, col] != 0 and last_empty and col < self.size[1] - 1 and buffer[row, col + 1] == 0:
                    filling = True
                last_empty = this_empty

    def _get_target(self, reward, agents):
        """
        Decide which map will the reward be filled to

        No effect on negative rewards. They will all go to map[0]
        """
        if reward <= 0:
            return [0]
        elif agents is None:
            return [i + 1 for i in range(self.num_agents)]
        else:
            return [i + 1 for i in agents]
