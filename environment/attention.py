import torch
import numpy as np

from utils import to_pixel
from .feature_map import FeatureMap


class AttentionMap(FeatureMap):
    """
    Attention map indicates every agents' position:
        map[i] is a heat map centered at agents[i].p
    """

    def __init__(self, device, h, w, num_agents):
        super(AttentionMap, self).__init__(device, h, w, num_agents)
        self.color = torch.tensor(
            [0, 1, 0], device=device, dtype=torch.float).view(3, 1)
        self.collide_color = torch.tensor(
            [1, 1, 0], device=device, dtype=torch.float).view(3, 1)
        self.done_color = torch.tensor(
            [0, 0, 1], device=device, dtype=torch.float).view(3, 1)
        self.is_normalized = True

    def update(self, centers: torch.Tensor):
        """
        Params:
            centers:    position of agents, torch of shape (n, 2)
        """
        self.map = self._make_gaussian(centers.float(), radius=64)

    def render(self, img, collided, done):
        """
        Add feature map contents to an image buffer
        """
        idx = torch.from_numpy(~done).to(self.device)
        idx = torch.nonzero((self.map[idx] > 0.999).sum(dim=0))
        img[:, idx[:, 0], idx[:, 1]] = self.color

        idx = torch.from_numpy(collided).to(self.device)
        idx = torch.nonzero((self.map[idx] > 0.999).sum(dim=0))
        img[:, idx[:, 0], idx[:, 1]] = self.collide_color

        idx = torch.from_numpy(done).to(self.device)
        idx = torch.nonzero((self.map[idx] > 0.999).sum(dim=0))
        img[:, idx[:, 0], idx[:, 1]] = self.done_color
