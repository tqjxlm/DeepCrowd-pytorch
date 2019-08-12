import torch
import numpy as np

from utils import to_pixel
from .feature_map import FeatureMap


class AttentionMap(FeatureMap):
    def __init__(self, device, h, w, num_agents):
        super(AttentionMap, self).__init__(device, h, w, num_agents)
        self.color = torch.tensor(
            [0, 1, 0], device=device, dtype=torch.float).view(3, 1)
        self.collide_color = torch.tensor(
            [0, 0, 1], device=device, dtype=torch.float).view(3, 1)
        self.is_normalized = True

    def update(self, centers: torch.Tensor):
        """
        Params:
            centers:    position of agents, torch of shape (n, 2)
        """
        new_map = self._make_gaussian(centers.float(), radius=128)
        valid = torch.all(centers >= 0, dim=1)
        new_map[valid].zero_()
        self.map = new_map

    def render(self, img, collided):
        """
        Add feature map contents to an image buffer

        It only renders the first agent's view

        parameters:
            img:    Tensor with shape (3, h, w)
        """
        idx = torch.from_numpy(~collided).to(self.device)
        idx = torch.nonzero((self.map[idx] > 0.999).sum(dim=0))
        img[:, idx[:, 0], idx[:, 1]] = self.color

        idx = torch.from_numpy(collided).to(self.device)
        idx = torch.nonzero((self.map[idx] > 0.999).sum(dim=0))
        img[:, idx[:, 0], idx[:, 1]] = self.collide_color
