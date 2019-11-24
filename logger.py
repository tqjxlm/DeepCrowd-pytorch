import sys
from collections import deque, namedtuple
from pprint import pprint

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from environment.env import Environment
from config import Config
from utils import StdoutAdaptor

Stat = namedtuple('Stat', ['success_rate', 'reward',
                           'collide', 'step', 'timeout', 'entropy'])

Loss = namedtuple('Loss', ['a_loss', 'c_loss', 'e_loss'])


class Logger:
    """
    A logging helper to manage all kinds of logging: stdout, file, tensorboard
    """

    def __init__(self, env: Environment, cfg: Config):
        self.training = cfg.training
        self.env = env

        # Redirect stdout
        print('Log syncing to', cfg.log_path)
        sys.stdout = StdoutAdaptor(cfg.log_path)

        print('----------------- Hyperparameters -----------------')
        pprint(vars(cfg))
        print('----------------- Hyperparameters -----------------')

        # Buffers
        self.buffer = np.zeros((cfg.maximum_step, 3, cfg.total_agents, 2))
        self.best_reward = -1000
        self.stats = deque([], 100)
        self.losses = deque([], 100)
        self.log_every = cfg.log_every
        self.reward = np.zeros((cfg.total_agents))

        if self.training:
            self.writer = SummaryWriter(
                log_dir='checkpoints/%s' % cfg.save_name)

        if cfg.output_name is None:
            self.output_path = None
        else:
            self.output_path = './output/%s.csv' % cfg.output_name

    def save_output_step(self, num_itr):
        if self.output_path is None:
            return

        self.buffer[num_itr][0] = self.env.agents.p
        self.buffer[num_itr][1] = self.env.agents.v
        self.buffer[num_itr][2] = self.env.agents.a

    def reset_stat(self):
        self.success_total = 0
        self.collide_total = 0
        self.reward_total = 0
        self.entropy_total = 0
        self.step_total = 0
        self.reward.fill(0)

    def record_stat(self, success, collide, reward, entropy, done):
        self.success_total += success
        self.collide_total += collide
        self.reward += reward.cpu().numpy()
        self.reward_total += reward.sum().item()
        self.entropy_total += (entropy.cpu().numpy() * (~done)).sum()
        self.step_total += (~done).sum()

    def finish_stat(self, cfg, done):
        return Stat(
            success_rate=self.success_total / cfg.total_agents * 100,
            reward=self.reward_total / cfg.total_agents,
            step=self.step_total / cfg.total_agents,
            collide=self.collide_total / cfg.total_agents,
            timeout=np.count_nonzero(~done),
            entropy=self.entropy_total / self.step_total
        )

    def log_episode(self, eps_local, stat, step, eps_global):
        if self.training:
            self.stats.append(stat)
            mean_stat = Stat(*tuple(sum(stat) / len(stat)
                                    for stat in zip(*self.stats)))

            # Logging to tensorboard
            self._write_tensorboard(eps_global, stat=stat, loss=None)

            # Logging to stdout
            if eps_local % self.log_every == 0:
                print('Episode %4d. Success %6.2f%%. Reward %5.2f. Collide %5.2f. Step %4.0f. Timeout %4.1f. Entropy %4.2f.' % (
                    eps_global, *mean_stat), end=' ')

            return mean_stat
        else:
            # If we get better performance, log it
            if (self.output_path is not None) and (stat.reward > self.best_reward):
                print('Get better reward %.3f, write to output file...' % stat.reward)
                self._write_output(step)
                self.best_reward = stat.reward

            # Logging to stdout
            print('Episode %4d. Success %6.2f%%. Reward %5.2f. Collide %5.2f. Step %4.0f. Timeout %4.1f. Entropy %4.2f' % (
                eps_local, *stat))
            return stat

    def log_loss(self, eps_local, loss, eps_global):
        self.losses.append(loss)
        mean_losses = [sum(loss) / len(loss) for loss in zip(*self.losses)]

        # Logging to tensorboard
        self._write_tensorboard(eps_global, stat=None, loss=loss)

        # Logging to stdout
        if eps_local % self.log_every == 0:
            print('Losses (a: %6.3f, c: %6.3f, e: %6.3f)' %
                  (*mean_losses, ))

    def _write_tensorboard(self, eps_global, stat=None, loss=None):
        if not self.training:
            return

        if stat is not None:
            self.writer.add_scalar('reward_agent_eps', stat.reward, eps_global)
            self.writer.add_scalar('success_rate_eps', stat.success_rate, eps_global)
            self.writer.add_scalar('step_agent_eps', stat.step, eps_global)
            self.writer.add_scalar('collision_agent_eps', stat.collide, eps_global)
            self.writer.add_scalar('entropy_agent_step', stat.entropy, eps_global)

        if loss is not None:
            self.writer.add_scalar('actor_loss', loss.a_loss, eps_global)
            self.writer.add_scalar('critic_loss', loss.c_loss, eps_global)
            self.writer.add_scalar('entropy_loss', loss.e_loss, eps_global)

    def _write_output(self, num_itr):
        if self.output_path is None:
            return

        with open(self.output_path, 'w') as output:
            output.write('step')
            for itr in range(num_itr):
                output.write('%d' % itr)
                p, v, a = self.buffer[itr][0], self.buffer[itr][1], self.buffer[itr][2]
                for i in range(self.env.num_agents):
                    output.write(',%.3f,%.3f' % (p[i][0], p[i][1]))
                    output.write(',%.3f,%.3f' % (v[i][0], v[i][1]))
                    output.write(',%.3f,%.3f' % (a[i][0], a[i][1]))
                output.write('\n')
