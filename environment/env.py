import random
import numpy as np
from collections import namedtuple
import itertools
from math import ceil

import torch
import torch.nn.functional as F

from config import Config
from utils import to_pixel, same_pixel, clip_length, Profiler, Agents, out_of_bound, generate_vector
from .feature_map import FeatureMap
from .obstacle import ObstacleMap, PolygonObstacle, CircleObstacle
from .crowd import CrowdMap
from .attention import AttentionMap
from .stage import get_stage, Stage


class Environment:

    def __init__(self, cfg: Config, device):
        self.cfg = cfg
        self.num_agents = cfg.total_agents
        self.size = np.array(cfg.stage_size)
        self.device = device

        h, w = cfg.stage_size
        self.crowd_map = CrowdMap(device, h, w, self.num_agents)
        self.obs_map = ObstacleMap(device, h, w, self.num_agents)
        self.attention_map = AttentionMap(device, h, w, self.num_agents)

        self.img = torch.zeros([3, h, w], device=device)

        self.stage = get_stage(self, cfg.stage)

        self.reset_reward()

    def share_memory(self):
        """
        Make all data support multi-process
        """

        self.crowd_map.share_memory()
        self.obs_map.share_memory()

    def render(self):
        """
        Render the current stage with all feature maps
        """
        with Profiler('render'):
            with torch.no_grad():
                self.img.zero_()
                self.obs_map.render(self.img)
                self.attention_map.render(self.img, self.collided, self.done)

                # (C, H, W) to (W, H, C) and flip Y axis for showing as an image
                return np.flip(self.img.cpu().numpy().transpose(2, 1, 0), axis=1)

    def reset_reward(self):
        """
        Reset the reward
        """
        self.stage.reset_stage()

    def reset(self):
        """
        Generate a number of agents randomly

        It must be called after the stage is set, otherwise agents may collide with obstacles

        Return:
            states:     initial states of all agents, size ((N, C, H, W), (N, C_l, H_l, W_l))
            done:       initial terminal flags, size (N, )
            hidden:     initial memory hidden feature, size (RNN_LAYER_SIZE, N, RNN_HIDDEN_SIZE)
        """
        self.agents = self.stage.reset_agents()

        # Initial states
        done = np.zeros((self.num_agents,), dtype=bool)
        self.success = np.zeros((self.num_agents,), dtype=bool)
        if self.cfg.rnn_type == 'LSTM':
            hidden = (
                torch.zeros(
                    [self.cfg.rnn_layer_size, self.num_agents, self.cfg.rnn_hidden_size], device=self.device),
                torch.zeros(
                    [self.cfg.rnn_layer_size, self.num_agents, self.cfg.rnn_hidden_size], device=self.device)
            )
        else:
            hidden = torch.zeros(
                [self.cfg.rnn_layer_size, self.num_agents, self.cfg.rnn_hidden_size], device=self.device)
        return self._generate_state(done), done, hidden

    def step(self, acceleration, pre_done, is_last_step):
        """
        Render a frame and move all agents.

        Should be called outside of any agent's action loop

        Params:
            acceleration:   numpy array of shape (N, 2), meaning direction and magnitude
            pre_done:       numpy array of shape (N, )

        Return:
            state:      Torch tensor of shape (N, C, H, W)
            reward:     Torch tensor of shape (N,)
            done:       numpy array of shape (N,)
            success:    success count
            collide:    inter-agent collide count
        """

        with Profiler('env step'):
            # Update velocity
            # acc_dir, acc_len = acceleration[:, 0], acceleration[:, 1]
            # self.agents.a = generate_vector(acc_dir * np.pi, acc_len[:, None] * self.cfg.max_a)
            self.agents.a = clip_length(acceleration, self.cfg.max_a)
            self.agents.v += self.agents.a
            self.agents.v = clip_length(self.agents.v, self.cfg.max_v)

            # Collision detection
            dest = self.agents.p + self.agents.v
            dest, reward, done, success, collide = self._move_agents(
                self.agents.p, dest, pre_done)

            self.agents.v[self.collided] = np.array([0, 0])

            # Timeout penalty
            if is_last_step:
                reward[~done] = self.cfg.fail_penalty

            # Effort penalty
            reward -= self.cfg.effort_ratio * np.linalg.norm(self.agents.v, axis=1) * ~pre_done
            reward *= self.cfg.reward_scale

            # Update position
            self.agents.p[~pre_done] = dest[~pre_done]
            new_p = to_pixel(self.agents.p[~pre_done])

        # Update crowd map
        with Profiler('crowd map'):
            self.crowd_map.map[0].zero_()
            self.crowd_map.map[0, new_p[:, 0], new_p[:, 1]] = 1
            self.crowd_map.update(self.agents.p, pre_done)

        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

        return self._generate_state(done), reward, done, success, collide

    def _move_agents(self, pos, end, pre_done):
        """
        Use a scan line method to move all the agents given their intended destination
        If there's any collision along the path, they will be stopped before the collision

        Input:
            pos:        starting point of agents. Numpy array of shape (N, 2)
            end:        intended end point of agents. Numpy array of shape (N, 2)
            pre_done:   previous status of agents. Numpy array of shape (N,)

        Return:
            end:        actual ending point after collision test. Numpy array of shape (N, 2)
            reward:     reward after collision test. Numpy array of shape (N, )
            done:       if state is done after collision test. Numpy array of shape (N, )
            success:    success count
            collide:    collide count
        """

        done = pre_done.copy()
        start = pos.copy()

        collided = np.zeros((self.num_agents,), dtype=bool)
        reward = np.zeros((self.num_agents,))
        success = np.zeros((self.num_agents,), dtype=bool)

        start_idx = to_pixel(start)
        end_idx = to_pixel(end)
        pxl_distance = end_idx - start_idx

        num_step = ceil(self.cfg.max_v_per_second)
        stride = (end - start) / num_step
        stride[done] = np.array([0, 0])

        obs_map = self.obs_map.cpu()[0]
        reward_map = self.obs_map.cpu()[1:]

        # March agents
        for _ in range(num_step):
            start += stride
            pxls = to_pixel(start)

            # Scan for collision
            # TODO: Maybe we can vectorize it
            for i in range(self.num_agents):
                if done[i] or collided[i]:
                    continue

                crash = False
                pxl = pxls[i]

                # Stage bound
                if out_of_bound(pxl, self.size):
                    crash = True
                    # collided[i] = True
                    # reward[i] = self.collide_penalty
                # Reward (the only terminal condition)
                elif reward_map[i, pxl[0], pxl[1]] > 0:
                    done[i] = True
                    success[i] = True
                    reward[i] = self.cfg.finish_reward
                # Obstacle
                elif obs_map[pxl[0], pxl[1]] > 0:
                    crash = True
                    # collided[i] = True
                    # reward[i] = -obs_map[pxl[0], pxl[1]]
                # Inter-agent collision
                else:
                    # Check against all unfinished agents
                    tmp = done[i]
                    done[i] = True
                    diff = pxls[np.logical_and(~done, ~collided)] - pxl
                    done[i] = tmp

                    for j in range(diff.shape[0]):
                        if diff[j][0] == 0 and diff[j][1] == 0:
                            collided[i] = True
                            break

                # If there is collision, push the agent back for one cell
                if collided[i] or crash:
                    if crash and self.cfg.terminate_on_crash:
                        done[i] = True
                        reward[i] = self.cfg.fail_penalty
                    else:
                        start[i] -= stride[i] / np.abs(stride[i]).max()
                        pxls[i] = to_pixel(start[i])
                        reward[i] = self.cfg.collide_penalty

            # Stop terminated or collided agents
            stride[done] = np.array([0, 0])
            stride[collided] = np.array([0, 0])

        self.collided = collided
        self.done = done
        self.success = np.logical_or(self.success, success)
        start = np.clip(start, [0, 0], [self.size[1]-1, self.size[0]-1])
        return start, reward, done, success.sum(), collided.sum()

    def _generate_state(self, done):
        """
        Generate states for one specific or all agents

        Return:
            full_map:       Batch of full feature maps. Torch tensor of shape (N, C, H, W)
            local_map:      Batch of local feature maps. Torch tensor of shape (N, C_local, H_local, W_local)
        """
        with Profiler('generate state'):
            # Feature tensor tells about the environment
            feature_tensor = torch.cat(
                (self.crowd_map.get_feature(), self.obs_map.get_feature()), dim=1)  # (N, 3, H, W)

            # Attention tensor indicates self position
            self.attention_map.update(torch.tensor(self.agents.p))
            attention_tensor = self.attention_map.get_feature().unsqueeze(1)  # (N, 1, H, W)

            full_map = torch.cat((attention_tensor, feature_tensor), dim=1)

            # Crop a local view feature for every agent
            local_map = self._crop_local_map(done, self.cfg.local_map_size, self.cfg.local_input_channel, feature_tensor)

        return full_map, local_map

    def _crop_local_map(self, done, size, local_input_channel, feature_tensor):
        """
        Crop a smaller part of the map centered on every agent position
        """

        local_map = torch.zeros(
            [self.num_agents, local_input_channel, size, size], device=self.device)

        # Pad all maps to fix out-of-bound conditions
        # TODO: There should be a faster way to do this
        pad = size // 2
        padding = (pad, pad, pad, pad)
        full_map_padded = F.pad(feature_tensor, padding, mode='replicate')

        # Crop a rect centered on agent positions
        for i in range(self.num_agents):
            if done[i]:
                continue

            center = to_pixel(self.agents.p[i])
            local_map[i] = full_map_padded[i][2:, center[0]:center[0]+size, center[1]:center[1]+size]

        return local_map
