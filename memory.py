import torch
import numpy as np

from config import Config

class Memory():
    """
    Rollout memory class

    No tensor in the memory requires gradient!

    The memory serves for the following benefits:
        * break the correlation among consecutive steps
        * allow sampling from previous experience
        * avoid gradient tracking during inference
    """

    def __init__(self, cfg: Config, device):
        self.train_unit = cfg.trainable_agents
        self.capacity = cfg.maximum_step
        self.device = device
        self.lstm = cfg.rnn_type == 'LSTM'
        # self.prioritized = cfg.prioritized_memory
        self.gamma = cfg.gamma

        capacity = cfg.maximum_step
        train_unit = cfg.trainable_agents
        self.full_feature = torch.zeros(
            [capacity, train_unit, cfg.global_input_channel, *cfg.stage_size], device=device)
        self.local_feature = torch.zeros(
            [capacity, train_unit, cfg.local_input_channel, cfg.local_map_size, cfg.local_map_size], device=device)
        self.actions = torch.zeros([capacity, train_unit, 2], device=device)
        self.rewards = torch.zeros([capacity, train_unit], device=device)
        self.probs = torch.zeros([capacity, train_unit], device=device)
        self.values = torch.zeros([capacity, train_unit], device=device)
        self.returns = torch.zeros([capacity, train_unit], device=device)
        self.advs = torch.zeros([capacity, train_unit], device=device)
        self.masks = torch.zeros(
            [capacity, train_unit], device=device, dtype=torch.int32)

        self.hidden = torch.zeros(
            [capacity, train_unit, cfg.rnn_layer_size, cfg.rnn_hidden_size], device=device)
        if self.lstm:
            self.cell = torch.zeros(
                [capacity, train_unit, cfg.rnn_layer_size, cfg.rnn_hidden_size], device=device)
        
        self.reset()

    def reset(self):
        self.pos = 0    # pos is alway the next (invalid) position
        self.masks.zero_()

    def insert(self, state, hidden, action, reward, prob, value, mask):
        """
        Params:
            state:      tuple (tensor of size (N, C, H, W), tensor of size (N, C_l, H_l, W_l))
            hidden:     tensor of size (rnn_layer_size, N, rnn_hidden_size) or a tuple of two
            action:     tensor of size (N, 2)
            reward:     tensor of size (N, )
            prob:       tensor of size (N, )
            value:      tensor of size (N, )
            mask:       tensor of size (N, )
        """

        self.full_feature[self.pos] = state[0]
        self.local_feature[self.pos] = state[1]
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.probs[self.pos] = prob
        self.values[self.pos] = value
        self.masks[self.pos] = mask

        if self.lstm:
            self.hidden[self.pos] = hidden[0].transpose(1, 0)
            self.cell[self.pos] = hidden[1].transpose(1, 0)
        else:
            self.hidden[self.pos] = hidden

        self.pos += 1

    def finish_rollout(self, next_v: torch.Tensor):
        """
        Calculate the advantage value used in optimization

        Params:
            next_v:     the next state value for all agents, tensor fo shape (N, )
        """
        self.size = self.pos
        pos = self.pos
        R = next_v
        while pos >= 1:
            pos -= 1
            R = self.gamma * R + self.rewards[pos] * self.masks[pos].float()
            self.returns[pos] = R.detach()
            self.advs[pos] = (R - self.values[pos]) * self.masks[pos].float()

    def sample(self, batch_size, success_rate, success):
        """
        Sample batches of history steps that cover all the memory

        Return:
            s:      states, tuple of ((B, C, H, W) tensor, (B, C_l, H_l, W_l) tensor)
            a:      actions, (B, 2) tensor
            h:      rnn hidden states, (rnn_layer_size, B, rnn_hidden_size) tensor
            p:      log probabilities, (B, ) tensor
            v:      predicted values, (B, ) tensor
            r:      true returns, (B, ) tensor
            adv:    advantage, (B, ) tensor
        """
        valid_agent = []
        valid_step = []

        # Balance positive and negative experience
        if success_rate == 0 or success_rate == 100:
            negative_scale = 1
            positive_scale = 1
        elif success_rate < 50:
            negative_scale = 1
            positive_scale = (int)(50 / success_rate) + 1
        else:
            # negative_scale = (int)(50 / (100-success_rate)) + 1
            negative_scale = 1
            positive_scale = 1

        for i in range(self.train_unit):
            if success[i]:
                duplicate = positive_scale
            else:
                duplicate = negative_scale
            for j in range(self.capacity):
                if self.masks[j, i] == 1:
                    for _ in range(duplicate):
                        valid_agent.append(i)
                        valid_step.append(j)

        valid_idx = np.array([valid_step, valid_agent]).T
        # if self.prioritized:
        #     priority = torch.log(self.returns[:self.size] - torch.min(self.returns[:self.size], dim=0).values)
        #     priority = (priority / priority.sum()).cpu().numpy()
        # else:
        np.random.shuffle(valid_idx)

        total_batch = len(valid_idx)
        for i in range(0, total_batch, batch_size):
            # if self.prioritized:
            #     idx = np.random.choice(valid_idx, batch_size, p=priority)
            # else:
            if i + batch_size > total_batch:
                idx = valid_idx[i:]
            else:
                idx = valid_idx[i: i + batch_size]

            full_feature = self.full_feature[idx[:, 0], idx[:, 1]]
            local_feature = self.local_feature[idx[:, 0], idx[:, 1]]
            actions = self.actions[idx[:, 0], idx[:, 1]]
            probs = self.probs[idx[:, 0], idx[:, 1]]
            values = self.values[idx[:, 0], idx[:, 1]]
            returns = self.returns[idx[:, 0], idx[:, 1]]
            advantages = self.advs[idx[:, 0], idx[:, 1]]
            hidden = self.hidden[idx[:, 0], idx[:, 1]].transpose(1, 0).contiguous()

            cnn_features = (full_feature, local_feature)
            if self.lstm:
                cell = self.cell[idx[:, 0], idx[:, 1]].transpose(1, 0).contiguous()
                rnn_features = (hidden, cell)
            else:
                rnn_features = hidden

            yield cnn_features, actions, rnn_features, probs, values, returns, advantages
