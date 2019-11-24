from itertools import count

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from model import Model
from memory import Memory
from environment.env import Environment

from logger import Logger, Stat, Loss
from utils import Profiler, init_weights
from config import Config


class Controller():
    """
    Training and controlling unit
    """

    def __init__(self, cfg: Config, env: Environment, device):
        # Save some config for convenience
        self.device = device
        self.sampling = cfg.sample_action
        self.cfg = cfg
        self.env = env
        self.global_episode = 1
        self.global_best = 0
        self.total_step = 0

        self.distribution = torch.distributions.Normal

        # Init model
        self._init_model()
        if not cfg.training:
            self.model.eval()

        # Setup training
        self.logger = Logger(env, cfg)

        if cfg.training:
            self.mem = Memory(cfg, device)
            self.chosen_agents = [i for i in range(cfg.trainable_agents)]

    def run_episode(self, i_episode, render_buffer: mp.Queue):
        """
        Simulate an episode from start to end
        """

        env = self.env
        cfg = self.cfg

        # New episode
        state, done, hidden = env.reset()
        self.logger.reset_stat()
        mean_stat = Stat(0,0,0,0,0,0)

        # Run until all agents done
        with Profiler('step'):
            for step in count():
                step_now = step

                # Terminal condition
                if np.all(done) or step == cfg.maximum_step:
                    self.total_step += step
                    break

                # Save values that should be inserted to memory before inference
                if cfg.training:
                    input_mask = torch.tensor(
                        (~done[self.chosen_agents]).astype(int), device=self.device)
                    input_state = (state[0][self.chosen_agents].detach(),
                                state[1][self.chosen_agents].detach())
                    if cfg.rnn_type == 'LSTM':
                        input_hidden = (
                            hidden[0][:, self.chosen_agents].detach(),
                            hidden[1][:, self.chosen_agents].detach()
                        )
                    else:
                        input_hidden = hidden[self.chosen_agents].detach()

                # Next step
                action, prob, value, hidden, entropy = self._act(state, hidden)

                self.logger.save_output_step(step)
                state, reward, done, success, collide = env.step(
                    action.cpu().numpy(), done, step == cfg.maximum_step - 1)

                # Save memory for training
                if cfg.training:
                    self.mem.insert(input_state, input_hidden, action[self.chosen_agents],
                                    reward[self.chosen_agents], prob[self.chosen_agents], value[self.chosen_agents], input_mask)

                if cfg.render and step % cfg.render_every == 0:
                    img = env.render()
                    render_buffer.put(img)

                # Bookkeeping
                self.logger.record_stat(success, collide, reward, entropy, done)

        # Finish episode
        stat = self.logger.finish_stat(cfg, done)

        # Generate a new stage once a while
        if i_episode % cfg.reset_stage_every == 0:
            env.reset_reward()

        if cfg.training:
            self.mem.finish_rollout(torch.zeros((cfg.trainable_agents), device=self.device))

            # Save checkpoints
            self._update_best(mean_stat.reward, cfg.save_name)
            if i_episode % cfg.save_every == 0:
                self._save_model(cfg.save_path)

            # Logging
            mean_stat = self.logger.log_episode(
                i_episode, stat, step_now, self.global_episode)

            # Optimize
            loss = Loss(*self._optimize(mean_stat.success_rate, env.success))

            # Finish logging
            self.logger.log_loss(i_episode, loss, self.global_episode)
            return mean_stat
        else:
            # Logging
            return self.logger.log_episode(i_episode, stat, step_now, self.global_episode)

    def _update_best(self, mean_reward, save_name):
        """
        Save the model whenever we get a better result
        """

        if mean_reward > self.global_best:
            self.global_best = mean_reward
            print('New best reward %.3f' % mean_reward, end=', ')
            self._save_model('checkpoints/%s_best.pt' % save_name)

    def _act(self, state, hidden):
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

    def _predict(self, state, hidden, action):
        """
        Predict state values given a state batch

        Params:
            state:      Batch of states. tuple (tensor (N, C, H, W), tensor (B, C_l, H_l, W_l))
            hidden:     Batch of hidden states. Tensor with shape (N, rnn_size)
            action:     Batch of predicted actions. Tensor with shape (N, 2)

        Return:
            value:      State value estimation. Tensor with shape (N, )
            prob:       Log probability of the given action. Tensor with shape (N, )
            entropy:    Entropy of the given action. Tensor with shape (1, )
        """

        mu, sigma, value, _ = self.model(state, hidden)
        m = self.distribution(mu, sigma)

        return m.log_prob(action).sum(dim=1), value.squeeze(1), m.entropy().sum(dim=1)

    def _optimize(self, success_rate, success):
        """
        Optimize the model using current memory
        """

        cfg = self.cfg
        gamma, clip, max_grad_norm, a_ratio, c_ratio, e_ratio, batch_size = (
            cfg.gamma, cfg.clip, cfg.max_grad_norm, cfg.actor_ratio, cfg.critic_ratio, cfg.entropy_ratio, cfg.batch_size)

        self.global_episode += 1

        if self.mem.size == 0:
            print('warning: empty memory')
            return [0, 0, 0]

        with Profiler('optimize'):
            total_loss = ([], [], [])
            b = 0
            for batch in self.mem.sample(batch_size, success_rate, success):
                # Importance sampling
                s, a, h, old_p, old_v, r, adv = batch
                p, v, e = self._predict(s, h, a)

                # Actor loss
                dist_ratio = torch.exp(p - old_p)
                a1 = adv * dist_ratio
                a2 = adv * torch.clamp(dist_ratio, 1.0 - clip, 1.0 + clip)
                a_loss = -torch.min(a1, a2).mean() * a_ratio

                # Critic loss
                v_clipped = old_v + (v - old_v).clamp(-clip, clip)
                v_loss_clipped = (v_clipped - r).pow(2)
                v_loss = (v - r).pow(2)
                c_loss = 0.5 * torch.max(v_loss, v_loss_clipped).mean() * c_ratio

                # Entropy
                e_loss = e.mean() * e_ratio

                self.optimizer.zero_grad()
                loss = a_loss + c_loss - e_loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                total_loss[0].append(a_loss.item())
                total_loss[1].append(c_loss.item())
                total_loss[2].append(e_loss.item())

            self.mem.reset()
            return tuple(sum(items) / len(items) for items in total_loss)

    def _init_model(self):
        """
        Load or initialize a model for training
        """

        cfg = self.cfg

        print('Initializing model...')
        self.model = Model(cfg)
        if cfg.load_path is not None:
            cp = torch.load(cfg.load_path,
                            map_location=None if cfg.cuda else self.device)
            self.model.load_state_dict(cp['model'])
            self.global_episode = cp['episode'] + 1

            if cfg.training:
                self.global_best = cp['best'] if cfg.save_name == cfg.load_name else -1000

        self.model.to(self.device)
        params = self.model.parameters()

        if cfg.training:
            self.optimizer = cfg.optimizer(
                params, lr=cfg.learning_rate, weight_decay=cfg.decay_rate)

            if cfg.load_path is not None:
                print('Loading checkpoint...')
                self.optimizer.load_state_dict(cp['opt'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                print('Checkpoint loaded')

    def _save_model(self, path):
        print('Saving checkpoint...')
        cp = {
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'episode': self.global_episode,
            'best': self.global_best
        }
        torch.save(cp, path)
        print('Checkpoint saved')
