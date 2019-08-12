import random
import numpy as np
from collections import namedtuple
import itertools

import torch
import torch.nn.functional as F

from config import Config
from utils import to_pixel, same_pixel, clip_length, Profiler, Agents, out_of_bound
from .feature_map import FeatureMap
from .obstacle import ObstacleMap, PolygonObstacle, CircleObstacle
from .crowd import CrowdMap
from .attention import AttentionMap
from .level import get_level, Level


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

        self.level = get_level(self, cfg.level)

        self.reset_reward()

    def share_memory(self):
        self.crowd_map.share_memory()
        self.obs_map.share_memory()

    def render(self):
        """
        Render the current stage with all feature maps
        """
        with torch.no_grad():
            self.img.zero_()
            self.obs_map.render(self.img)
            self.attention_map.render(self.img, self.collided)

            # (C, H, W) to (W, H, C) and flip Y axis for showing as an image
            return np.flip(self.img.cpu().numpy().transpose(2, 1, 0), axis=1)

    def generate_state(self, done):
        """
        Generate states for one specific or all agents

        Return:
            full_map:       Batch of full feature maps. Torch tensor of shape (N, C, H, W)
            local_map:      Batch of local feature maps. Torch tensor of shape (N, C_l, H_l, W_l)
        """
        with Profiler('generate state'):
            crowd_tensor = torch.cat(
                (self.crowd_map.get_feature(), self.obs_map.get_feature()), dim=1)  # (N, 3, H, W)

            self.attention_map.update(torch.tensor(self.agents.p))
            attention_tensor = self.attention_map.get_feature().unsqueeze(1)  # (N, 1, H, W)

            full_map = torch.cat((attention_tensor, crowd_tensor), dim=1)

            # crop a local view feature for every agent
            size = self.cfg.local_map_size
            local_map = torch.zeros(
                [self.num_agents, self.cfg.local_input_channel, size, size], device=self.device)
            pad = size // 2
            full_map_padded = F.pad(
                full_map, (pad, pad, pad, pad), mode='replicate')
            crowd_map_padded = F.pad(self.crowd_map.get_global_feature(
            ), (pad, pad, pad, pad), mode='replicate').squeeze()
            for i in range(self.num_agents):
                center = to_pixel(self.agents.p[i])
                try:
                    local_map[i, 0] = crowd_map_padded[center[0]
                        :center[0]+size, center[1]:center[1]+size]
                    local_map[i, 1:] = full_map_padded[i][2:, center[0]
                        :center[0]+size, center[1]:center[1]+size]
                except:
                    print(center)
                    print(self.agents.p[i])
                    raise NotImplementedError

        return full_map, local_map

    def reset_reward(self):
        """
        Reset the reward
        """
        self.level.reset_level()

    def reset(self):
        """
        Generate a number of agents randomly

        It must be called after the stage is set, otherwise agents may collide with obstacles
        """
        self.agents = self.level.reset_agents()

        # Initial states
        done = np.zeros((self.num_agents,), dtype=bool)
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
        return self.generate_state(done), done, hidden

    def step(self, acceleration, pre_done):
        """
        Render a frame and move all agents.

        Should be called outside of any agent's action loop

        Params:
            acceleration:   numpy array of shape (N, 2)
            pre_done:       numpy array of shape (N, )

        Return:
            state:      Torch tensor of shape (N, C, H, W)
            reward:     Torch tensor of shape (N,)
            done:       numpy array of shape (N,)
            success:    success count
            collide:    inter-agent collide count
        """
        with Profiler('step'):
            # Update velocity
            self.agents.a = clip_length(acceleration, self.cfg.max_a)
            self.agents.v += self.agents.a
            self.agents.v = clip_length(self.agents.v, self.cfg.max_v)

            # Collision detection
            old_p = to_pixel(self.agents.p)
            dest = self.agents.p + self.agents.v
            dest, reward, done, success, collide = self.check_collision(
                self.agents.p, dest, pre_done)

            # Effort penalty
            reward -= self.cfg.effort_ratio * \
                (np.linalg.norm(self.agents.a, axis=1) + 1) * ~done

            reward *= self.cfg.reward_scale

            # Update position
            self.agents.p[~pre_done] = dest[~pre_done]
            new_p = to_pixel(self.agents.p[~pre_done])

        # Update crowd map
        with Profiler('update crowd map'):
            self.crowd_map.map[0, old_p[:, 0], old_p[:, 1]] = 0
            self.crowd_map.map[0, new_p[:, 0], new_p[:, 1]] = 1
            self.crowd_map.update(self.agents.p, done)

        return self.generate_state(done), torch.tensor(reward, dtype=torch.float32, device=self.device), done, success, collide

    def check_collision(self, pos, end, pre_done):
        """
        Input:
            pos:        starting point of agents. Numpy array of shape (N, 2)
            end:        intended end point of agents. Numpy array of shape (N, 2)
            pre_done:   previous status of agents. Numpy array of shape (N,)

        Return:
            end:        actual ending point after collision test. Numpy array of shape (N, 2)
            reward:     reward after collision test. Numpy array of shape (N, )
            done:       if state is done after collision test. Numpy array of shape (N, )
            success:    success count
            agent_collide:      inter-agent collide count
        """
        done = pre_done.copy()
        # stopped = done.copy()
        start = pos.copy()
        collided = np.zeros((self.num_agents,), dtype=bool)

        reward = np.zeros((self.num_agents,))
        success = 0

        start_idx = to_pixel(start)
        end_idx = to_pixel(end)
        pxl_distance = end_idx - start_idx

        num_step = max(np.max(np.abs(pxl_distance[~done])), 1)
        stride = (end - start) / num_step
        stride[done] = np.array([0, 0])

        # March agents
        for _ in range(num_step):
            start += stride
            pxls = to_pixel(start)

            # Scan for collision
            # TODO: extremely slow this way. better vectorize it
            for i in range(self.num_agents):
                if done[i] or collided[i]:
                    continue
                pxl = pxls[i]
                push_back = False

                # Stage collision
                if out_of_bound(pxl, self.size):
                    done[i] = True
                    reward[i] = -10
                    push_back = True

                # Obstacle collision
                elif self.obs_map.map[0, pxl[0], pxl[1]] > 0:
                    done[i] = True
                    reward[i] = -self.obs_map.map[0, pxl[0], pxl[1]]
                    push_back = True

                # Reward collision
                elif self.obs_map.map[i+1, pxl[0], pxl[1]] > 0:
                    done[i] = True
                    success += 1
                    reward[i] = self.obs_map.map[i+1, pxl[0], pxl[1]]

                # Inter-agent collision
                else:
                    for j in range(self.num_agents):
                        # For any other running agents, check collision
                        # Episodes are not terminated after collision, so agents should be pushed back and stopped
                        if not done[j] and not collided[j] and j != i and same_pixel(pxl, pxls[j]):
                            collided[i] = True
                            push_back = True
                            reward[i] = -10
                            break
                
                if push_back:
                    start[i] -= stride[i] / np.abs(stride[i]).max()
                    start[i] = np.clip(start[i], [0, 0], [self.size[1]-1, self.size[0]-1])

            # Stop collided agents
            # stopped[done] = True
            stride[done] = np.array([0, 0])
            stride[collided] = np.array([0, 0])

        self.collided = collided
        return start, reward, done, success, collided.sum()
