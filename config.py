import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import argparse
import json


class Config():
    """
    The hyperparameter class
    """

    def __init__(self):
        self.deterministic = True
        self.random_seed = 3
        self.device = 'cuda'
        self.cuda = True
        self.total_episode = 3000

        # Environment
        self.stage_size = (256, 256)
        self.render = False
        self.render_every = 1       # render interval
        self.reset_stage_every = 10  # reset stage interval
        # number of agents in the simulation (may not all be used in optimization)
        self.total_agents = 8
        self.maximum_step = 200     # maximum step number before the episode terminates
        self.step_per_second = 1
        self.max_v_per_second = 1.5    # maximum possible velocity for any agent
        self.max_a_per_second = 1.0      # maximum possible acceleration for any agent
        self.max_a = self.max_a_per_second / self.step_per_second
        self.max_v = self.max_v_per_second / self.step_per_second
        self.effort_ratio = 0          # penalty ratio for agents to make acceleration
        self.reward_scale = 0.1
        self.finish_reward = 10
        self.fail_penalty = -10
        self.reward_size = 1
        self.collide_penalty = -1
        # whether to do sampling when making actions or to be deterministic
        self.sample_action = True
        self.terminate_on_crash = True

        # Save / load / logging
        self.load_path = None
        self.output_name = None
        self.training = True       # whether do optimization and dump the model
        self.profiling = True
        self.save_every = 100
        self.log_every = 10

        # Model
        self.global_input_channel = 4      # number of feature maps
        self.local_input_channel = 3      # number of feature maps
        self.rnn_type = 'LSTM'      # must be 'LSTM' or 'GRU'
        self.activation = F.relu    # activation used in convolution and hidden layers
        base_channel = 8
        self.global_conv_setting = [
            (self.global_input_channel, base_channel * 1, 7, 4),
            (base_channel * 1, base_channel * 1, 3, 1),
            (base_channel * 1, base_channel * 2, 3, 2),
            (base_channel * 2, base_channel * 2, 3, 1),
            (base_channel * 2, base_channel * 4, 3, 2),
            (base_channel * 4, base_channel * 4, 3, 1)
        ]  # (in_ch, out_ch, kernel_size, stride)

        self.local_conv_setting = [
            (self.local_input_channel, base_channel, 3, 2),
            (base_channel * 1, base_channel * 1, 3, 1),
            (base_channel * 1, base_channel * 1, 3, 1)
        ]  # (in_ch, out_ch, kernel_size, stride)

    def parse_arguments(self):
        """
        Run time argument 
        """
        examples = '''example usage, you should modify ./configs/*.json to your needs:

        start a new train:                              python main.py train-initial
        train on an existing checkpoint:                python main.py train-resume
        run several inference and save outputs:         python main.py inference
        '''

        # Load a json config
        parser = argparse.ArgumentParser(description='DeepCrowd',
                                         epilog=examples,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)

        parser.add_argument('config_name', metavar='config_name', type=str,
                            help='the config name under ./configs/')

        parser.add_argument('challenge_name', metavar='challenge_name', type=str,
                            help='the challenge name under ./challenges/')

        args = parser.parse_args()
        if args.config_name is None:
            raise 'Error: you must provide a valid config name'

        if args.challenge_name is None:
            raise 'Error: you must provide a valid challenge name'

        challenge_path = './challenges/%s.json' % args.challenge_name
        try:
            clg_file = open(challenge_path)
            challenge = json.load(clg_file)
        except:
            raise 'Error: challenge file %s not valid' % challenge_path

        config_path = './configs/%s.json' % args.config_name
        try:
            cfg_file = open(config_path)
            cfg = json.load(cfg_file)
        except:
            raise 'Error: config file %s not valid' % config_path

        # Load challenge settings (environment)
        for key, value in challenge.items():
            setattr(self, key, value)

        # Load procedural parameters
        if 'gpu_id' in cfg:
            self.device = 'cuda:%d' % cfg['gpu_id']
        if 'episode' in cfg:
            self.total_episode = cfg['episode']
        if 'render' in cfg:
            self.render = cfg['render']
        if 'deterministic' in cfg:
            self.deterministic = cfg['deterministic']
        if 'sample_action' in cfg:
            self.sample_action = cfg['sample_action']

        if 'load' in cfg:
            self.load_name = cfg['load']
            if 'best' in cfg and cfg['best']:
                self.load_path = 'checkpoints/%s_best.pt' % self.load_name
            else:
                self.load_path = 'checkpoints/%s.pt' % self.load_name

        # Load modes
        if 'mode' not in cfg:
            raise 'Error: you must provide a mode in config'
        elif cfg['mode'] == 'inference':
            print('Mode: Inference')
            self.training = False

            if cfg['output'] is not None and cfg['output']:
                self.output_name = self.load_name
            self.log_path = 'checkpoints/%s_test.log' % self.load_name
        elif cfg['mode'] == 'train':
            print('Mode: Training')
            self.training = True

            if cfg['save'] is not None:
                self.save_name = cfg['save']
            else:
                self.save_name = 'temp'
                print('Warning: no save path specified, saving to "temp"')
            self.save_path = 'checkpoints/%s.pt' % self.save_name
            self.log_path = 'checkpoints/%s_train.log' % self.save_name

            # self.prioritized_memory = True
            self.optimizer = optim.Adam
            self.trainable_agents = 8       # number of agents used in optimization

            # Load training hyper-parameters
            if 'train' in cfg:
                for key, value in cfg['train'].items():
                    setattr(self, key, value)
        else:
            raise 'Error: invalid mode'

        # Load model hyper-parameters
        if 'model' in cfg:
            for key, value in cfg['model'].items():
                setattr(self, key, value)
        
        # Overwrite challenge settings
        if 'challenge' in cfg:
            for key, value in cfg['challenge'].items():
                setattr(self, key, value)

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
