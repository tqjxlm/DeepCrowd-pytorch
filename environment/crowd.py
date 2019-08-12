import torch
import numpy as np

from utils import to_pixel
from .feature_map import FeatureMap


class CrowdMap(FeatureMap):
    """
    Crowd map consists of two maps:
        map[0]  binary map that indicates crowd centers
        map[1]  heat map with normal distributions around crowd centers
    """

    def __init__(self, device, h, w, num_agents):
        super(CrowdMap, self).__init__(device, h, w, num_agents + 1)
        self.color = torch.tensor(
            [0, 1, 0], device=device, dtype=torch.float).view(3, 1)
        self.normalized = True
        self.num_agents = num_agents
        self.global_map = torch.zeros([1, h, w], device=device, dtype=torch.float32)

    def render(self, img):
        """
        Add feature map contents to an image buffer

        It only renders the first agent's view

        parameters:
            img:    Tensor with shape (3, h, w)
        """
        idx = torch.nonzero(self.map[1] > 0.99)
        img[:, idx[:, 0], idx[:, 1]] = self.color

    def update(self, centers, done):
        """
        Generate normal distributions around crowd centers, excluding the agent itself

        Parameters:
            centers:    Numpy array of shape (N, 2)
            done:       Indicates whether agents are still valid. Numpy array of shape (N, )
        """
        valid = np.nonzero(~done)[0]
        for agent_id in range(self.num_agents):
            excluded = centers[np.nonzero(valid != agent_id)[0]]
            self.map[agent_id+1] = self._make_gaussian(torch.from_numpy(excluded).float(), radius=32).sum(dim=0)
        
        self.global_map[0] = self._make_gaussian(torch.from_numpy(centers[valid]).float(), radius=3).sum(dim=0)

    def get_feature(self):
        """
        Return a crowd map for each agent

        Return:
            crowd_maps:     tensor of shape (N, 1, H, W)
        """
        return self.map[1:].unsqueeze(1)

    def get_global_feature(self):
        """
        Return a global crowd map with all agents

        Return:
            global_map:     tensor of shape (1, 1, H, W)
        """
        return self.global_map.unsqueeze(1)
