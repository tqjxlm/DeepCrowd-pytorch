import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from model import Model
from memory import Memory
from utils import Profiler, init_weights
from config import Config

class Controller():
    """
    Training and controlling unit
    """

    def __init__(self, cfg: Config, device):
        self.model = Model(cfg)
        self.device = device
        self.gamma = cfg.gamma
        self.clip = cfg.clip
        self.max_grad_norm = cfg.max_grad_norm
        self.a_ratio = cfg.actor_ratio
        self.c_ratio = cfg.critic_ratio
        self.e_ratio = cfg.entropy_ratio
        self.global_episode = 0
        self.global_best = 0
        self.sampling = cfg.sample_action

        self.distribution = torch.distributions.Normal

        if cfg.load_path is not None:
            cp = torch.load(cfg.load_path,
                            map_location=None if cfg.cuda else device)
            self.model.load_state_dict(cp['model'])
            self.global_episode = cp['episode']

            if cfg.training:
                self.global_best = cp['best'] if cfg.save_name == cfg.load_name else -1000
        # else:
        #     self.model.apply(init_weights)

        self.model.to(device)
        self.optimizer = cfg.optimizer(self.model.parameters(
        ), lr=cfg.learning_rate, weight_decay=cfg.decay_rate)

        if cfg.load_path is not None:
            self.optimizer.load_state_dict(cp['opt'])
            print('checkpoint loaded...')

        if not cfg.training:
            self.model.eval()

    def update_best(self, mean_reward, save_name):
        if mean_reward > self.global_best:
            self.global_best = mean_reward
            print('new best %.2f' % mean_reward, end=', ')
            self.save('checkpoints/%s_best.pt' % save_name)

    def act(self, state, hidden):
        """
        Select action given a state batch

        Params:
            state:      Batch of states. tuple (tensor (N, C, H, W), tensor (B, C_l, H_l, W_l))
            hidden:     Batch of hidden states. Tensor with shape (N, rnn_size)

        Return:
            action:     Acceleration. Tensor with shape (N, 2)
            prob:       Action probability. Tensor with shape (N, )
            value:      State value estimation. Tensor with shape (N, )
            hidden:     The new hidden states. Tensor with shape (N, rnn_size)
            entropy:    The current action entropy. Tensor with shape (N, )
        """
        with torch.no_grad():
            with Profiler('forward'):
                mu, sigma, value, h = self.model(state, hidden)
                m = self.distribution(mu, sigma)
                if self.sampling:
                    action = m.sample()
                else:
                    action = m.mean
                return action, m.log_prob(action).sum(dim=1), value.squeeze(1), h, m.entropy().sum(dim=1)

    def evaluate(self, state, hidden, action):
        """
        Predict value given a state batch

        Params:
            state:      Batch of states. tuple (tensor (N, C, H, W), tensor (B, C_l, H_l, W_l))
            hidden:     Batch of hidden states. Tensor with shape (N, rnn_size)

        Return:
            value:      State value estimation. Tensor with shape (N, )
            prob:       Log probability of the given action. Tensor with shape (N, )
            entropy:    Entropy of the given action. Tensor with shape (1, )
        """
        mu, sigma, value, _ = self.model(state, hidden)
        m = self.distribution(mu, sigma)
        
        return m.log_prob(action).sum(dim=1), value.squeeze(1), m.entropy().sum(dim=1)

    def optimize(self, mem, batch_size):
        """
        Params:
            mem:        memory class
            batch_size: batch size
        """
        self.global_episode += 1

        if mem.size == 0:
            print('warning: empty memory')
            return [0, 0, 0]

        with Profiler('optimize'):
            total_loss = ([], [], [])
            b = 0
            for batch in mem.sample(batch_size):
                # Importance sampling
                s, a, h, old_p, old_v, r, adv = batch
                p, v, e = self.evaluate(s, h, a)

                # Actor loss
                dist_ratio = torch.exp(p - old_p)
                a1 = adv * dist_ratio
                a2 = adv * torch.clamp(dist_ratio, 1.0 -
                                       self.clip, 1.0 + self.clip)
                a_loss = -torch.min(a1, a2).mean() * self.a_ratio

                # Critic loss
                v_clipped = old_v + (v - old_v).clamp(-self.clip, self.clip)
                v_loss_clipped = (v_clipped - r).pow(2)
                v_loss = (v - r).pow(2)
                c_loss = 0.5 * \
                    torch.max(v_loss, v_loss_clipped).mean() * self.c_ratio

                # Entropy
                e_loss = e.mean() * self.e_ratio

                self.optimizer.zero_grad()
                loss = a_loss + c_loss - e_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                total_loss[0].append(a_loss.item())
                total_loss[1].append(c_loss.item())
                total_loss[2].append(e_loss.item())

            mem.reset()
            return tuple(sum(items) / len(items) for items in total_loss)

    def save(self, path):
        cp = {
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'episode': self.global_episode,
            'best': self.global_best
            }
        torch.save(cp, path)
        print('checkpoint saved...')
