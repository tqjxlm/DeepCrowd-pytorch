import torch
import numpy as np


class FeatureMap():
    def __init__(self, device, h, w, c=1):
        self.map = torch.zeros([c, h, w], device=device, dtype=torch.float32)
        self.size = np.array((h, w))
        self.device = device
        self.color = torch.tensor(
            [0, 0, 1], device=device, dtype=torch.float).view(3, 1)
        self.is_normalized = False

    def reset(self, axis=None):
        if axis is None:
            self.map.zero_()
        else:
            self.map[axis].zero_()

    def render(self, img):
        """
        Add feature map contents to an image buffer

        parameters:
            img:    Tensor with shape (3, h, w)
            color:  Tensor with shape (3, 1, 1)
        """
        idx = torch.nonzero(self.map[0])
        img[:, idx[:, 0], idx[:, 1]] = self.color

    def _normalized(self):
        """
        Return a normalized copy of the feature map
        """
        return self.map / torch.max(self.map)

    def get_feature(self):
        """
        Return the feature map, size (C, H, W)
        """
        if not self.is_normalized:
            return self._normalized()
        else:
            return self.map

    def share_memory(self):
        self.map.share_memory()

    def _make_gaussian(self, centers: torch.Tensor, radius=32):
        """
        Make a square gaussian kernel.

        Params:
            centers:     tensor of shape (N, 2)
            radius:      radius of the kernel, scalar

        Return:
            heatmap:    a batch of heat maps, tensor shape (N, H, W)
        """

        y = torch.arange(0, self.size[0], 1, device=self.device)[:, None]
        x = torch.arange(0, self.size[1], 1, device=self.device)[None, :]

        centers[:, 0].clamp_(0, self.size[0] - 1)
        centers[:, 1].clamp_(0, self.size[1] - 1)
        y0 = centers[:, 0, None, None].to(device=self.device)
        x0 = centers[:, 1, None, None].to(device=self.device)

        return torch.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / radius ** 2)
