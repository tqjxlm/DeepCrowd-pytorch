import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import argparse


class Config():
    """
    The hyperparameter class
    """

    def __init__(self):
        self.deterministic = True
        self.random_seed = 3
        self.device = 'cuda'
        self.cuda = True

        # Environment
        self.stage_size = (256, 256)
        self.render = False
        self.render_every = 1       # render interval
        self.reset_stage_every = 10  # reset stage interval
        self.train_agents = 8       # number of agents that used in optimization
        # number of agents in the simulation (may not all be used in optimization)
        self.total_agents = 8
        self.max_a = 0.5            # maximum possible acceleration for any agent
        self.max_v = 2              # maximum possible velocity for any agent
        self.effort_ratio = 0.005   # penalty ratio for agents to make acceleration
        self.maximum_step = 1000    # maximum step number before the episode terminates
        self.reward_scale = 0.1
        self.fixed_reset_interval = True
        # whether to do sampling when making actions or to be deterministic
        self.sample_action = True

        # Save / load / Logging
        self.load_path = None
        self.output_name = None
        self.resume = False         # whether to load a checkpoint with the same name
        self.training = True       # whether do optimization and dump the model
        self.profiling = True
        self.save_every = 100
        self.log_every = 10

        # Train
        self.total_episode = 3000
        self.batch_size = 64        # optimization batch size after each episode
        self.gamma = 0.99
        self.learning_rate = 3e-6
        self.decay_rate = 0         # optimizer weight decay (l2 norm)
        self.clip = 0.2             # ppo loss clip threshold
        self.max_grad_norm = 0.5    # parameter clip threshold
        self.optimizer = optim.Adam
        self.actor_ratio = 1
        self.critic_ratio = 0.2
        # provide a positive value to encourage exploring, and a negative one to encourage determination
        self.entropy_ratio = 0
        self.master_threshold = 0.8
        self.master_time = 3
        self.prioritized_memory = False
        # target mean successful rate that will trigger early stopping
        self.early_stop = None

        # Model
        self.global_input_channel = 4      # number of feature maps
        self.local_input_channel = 3      # number of feature maps
        self.local_map_size = 17     # a finer detailed map centered at the agent location
        self.hidden_size = 256      # hidden layer size between cnn and final output
        self.rnn_hidden_size = 256  # recurrent layer hidden state feature size
        self.rnn_layer_size = 1     # recurrent layer number
        self.rnn_type = 'LSTM'      # must be 'LSTM' or 'GRU'
        self.activation = F.relu    # activation used in convolution and hidden layers
        self.dropout = 0
        base_channel = 8
        self.global_conv_setting = [
            (self.global_input_channel, base_channel, 7, 4),
            (base_channel * 1, base_channel * 1, 3, 1),
            (base_channel * 1, base_channel * 2, 3, 2),
            (base_channel * 2, base_channel * 2, 3, 1),
            (base_channel * 2, base_channel * 4, 3, 2),
            (base_channel * 4, base_channel * 4, 3, 1)
            # ,(64, 128, 3, 2)
            # ,(128, 128, 3, 1)
        ]  # (in_ch, out_ch, kernel_size, stride)

        self.local_conv_setting = [
            (self.global_input_channel, 64, 3, 2),
            (64, 64, 3, 1)
        ]  # (in_ch, out_ch, kernel_size, stride)

    def parse_arguments(self):
        """
        Run time argument 
        """
        examples = '''example usage:

        a default train on all gpu:       python main.py --level 0
        a default train on gpu#1:         python main.py --level 0 --gpu-id 1
        train for 10000 episodes:         python main.py --level 0 --episode 10000
        resume an existing train:         python main.py --level 0 --load checkpoint_name
        resume the last best record:      python main.py --level 0 --load checkpoint_name --best
        save to a custom name:            python main.py --level 0 --save checkpoint_name
        load and train next stage:        python main.py --level 1 -l cp_name --best -s another_cp_name
        test and render a checkpoint:     python main.py --level 1 --inference --render --load checkpoint_name --best
        '''

        parser = argparse.ArgumentParser(description='DeepCrowd',
                                         epilog=examples,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)

        parser.add_argument('--level', type=int,
                            help='the stage scenario number. required (available 0, 1, 2)')
        parser.add_argument('-g', '--gpu-id', type=int,
                            help='which gpu to use. (default: managed by pytorch)')
        parser.add_argument('-e', '--episode', type=int,
                            help='total episode to run. (default: 3000)')
        parser.add_argument('-a', '--agents', type=int,
                            help='total agent to simulate. (default: 8)')
        parser.add_argument('-l', '--load', type=str,
                            help='checkpoint to load, must be under ./checkpoints. (default: None)')
        parser.add_argument('-s', '--save', type=str,
                            help='checkpoint name to save, relative to ./checkpoints. (default: "cp")')
        parser.add_argument('-o', '--output', type=str,
                            help='inference result output name, relative to ./output. (default: load name)')

        parser.add_argument('-i', '--inference', action='store_true',
                            help='set to do inference only, without memory, optimization or saving checkpoints')
        parser.add_argument('-r', '--render', action='store_true',
                            help='set to render the stage in graphics. requires display availability')
        parser.add_argument('--no-deterministic', action='store_true',
                            help='set to disable deterministic random processes')
        parser.add_argument('--no-sampling', action='store_true',
                            help='set to disable action sampling. automatically set if in inference mode')
        parser.add_argument('--best', action='store_true',
                            help='set to load the best record of the given load name')

        args = parser.parse_args()
        if args.level is None:
            raise 'Error: you must provide a level number, for example --level 0'
        else:
            self.level = args.level

        if args.gpu_id is not None:
            self.device = 'cuda:%d' % args.gpu_id
        if args.episode is not None:
            self.total_episode = args.episode

        if args.load is not None:
            self.load_name = args.load
            if args.best:
                self.load_path = 'checkpoints/%s_best.pt' % self.load_name
            else:
                self.load_path = 'checkpoints/%s.pt' % self.load_name

        if args.inference:
            print('Notice: Inference mode')
            self.training = not args.inference
            if args.output is not None:
                self.output_name = args.output
            elif self.load_name is not None:
                self.output_name = self.load_name
        else:
            if args.save is not None:
                self.save_name = args.save
            else:
                self.save_name = 'temp'
                print('Warning: no save path specified, saving to "temp"')
            self.save_path = 'checkpoints/%s.pt' % self.save_name
            self.log_path = 'checkpoints/%s.log' % self.save_name

        if args.agents is not None:
            self.total_agents = args.agents

        if args.render:
            print('render enabled')
            self.render = args.render

        if args.no_deterministic:
            self.deterministic = False

        if args.no_sampling or args.inference:
            self.sample_action = False

    def global_deterministic(self):
        """
        Force all random processes to perform deterministically to ensure reproducibility

        Notice: this function is applied per process
        """
        if self.deterministic:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print('All processes set to deterministic')
