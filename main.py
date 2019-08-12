import sys
import signal
from collections import deque, namedtuple
from itertools import count
from pprint import pprint
import csv

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from pyqtgraph.Qt import QtGui, QtCore

from config import Config
from controller import Controller
from environment.env import Environment
from viewer import Viewer
from memory import Memory
from utils import Profiler, Logger

Stat = namedtuple('Stat',
                  ['success_rate', 'reward', 'collide', 'step', 'timeout', 'entropy'])

Loss = namedtuple('Loss',
                  ['a_loss', 'c_loss', 'e_loss'])


def write_output(name, buffer, env, num_itr):
    with open('./output/%s.csv' % name, 'w') as output:
        for itr in range(num_itr):
            output.write('%d' % itr)
            for i in range(env.num_agents):
                output.write(',%.3f,%.3f' % (buffer[itr][0][i][0],buffer[itr][0][i][1]))
                output.write(',%.3f,%.3f' % (buffer[itr][1][i][0],buffer[itr][1][i][1]))
                output.write(',%.3f,%.3f' % (buffer[itr][2][i][0],buffer[itr][2][i][1]))
            output.write('\n')

def write_output_buffer(buffer, env, num_itr):
    if buffer is None:
        return
    
    buffer[num_itr][0] = env.agents.p
    buffer[num_itr][1] = env.agents.v
    buffer[num_itr][2] = env.agents.a


def train_process(cfg: Config, render_buffer: mp.Queue):
    """
    Main train loop
    """
    # init running environment
    cfg.global_deterministic()

    if cfg.training:
        print('Log syncing to', cfg.log_path)
        sys.stdout = Logger(cfg.log_path)

    print('----------------- Hyperparameters -----------------')
    pprint(vars(cfg))
    print('----------------- Hyperparameters -----------------')

    device = torch.device(cfg.device)
    Profiler.enable_profiling(cfg.profiling)

    # init training
    env = Environment(cfg, device)
    ctrl = Controller(cfg, device)
    if cfg.training:
        writer = SummaryWriter(log_dir='checkpoints/%s' % cfg.save_name)
        mem = Memory(cfg, device)

    stats = deque([], 100)
    losses = deque([], 100)

    total_step = 0
    stage_episode = 0
    success_time = 0

    chosen_agents = [i for i in range(cfg.train_agents)]

    if cfg.output_name is not None:
        output_buffer = np.zeros((cfg.maximum_step, 3, cfg.total_agents, 2))
        best_reward = -1000
    else:
        output_buffer = None

    for i_episode in range(1, cfg.total_episode + 1):
        # New episode
        state, done, hidden = env.reset()
        write_output_buffer(output_buffer, env, 0)
        success_total = 0
        collide_total = 0
        reward_total = 0
        entropy_total = 0
        step_total = 0

        # Run an episode until all agents done
        for step in count():
            step_now = step
            if np.all(done) or step == cfg.maximum_step:
                if cfg.training:
                    mem.finish_rollout(torch.zeros(
                        (cfg.train_agents), device=device))
                total_step += step
                break

            # Save values that should be inserted to memory before inference
            input_mask = torch.tensor(
                (~done[chosen_agents]).astype(int), device=device)
            input_state = (state[0][chosen_agents].detach(), state[1][chosen_agents].detach())
            if cfg.rnn_type == 'LSTM':
                input_hidden = (
                    hidden[0][:, chosen_agents].detach(),
                    hidden[1][:, chosen_agents].detach()
                )
            else:
                input_hidden = hidden[chosen_agents].detach()

            # Step
            action, prob, value, hidden, entropy = ctrl.act(state, hidden)
            state, reward, done, success, collide = env.step(
                action.cpu().numpy(), done)
            write_output_buffer(output_buffer, env, step + 1)

            if cfg.training:
                mem.insert(input_state, input_hidden, action[chosen_agents],
                        reward[chosen_agents], prob[chosen_agents], value[chosen_agents], input_mask)

            if cfg.render and step % cfg.render_every == 0:
                img = env.render()
                # local = np.flip(state[1][0].cpu().numpy().transpose(2, 1, 0), axis=1)
                # img[:7, :7, 0] = local[:, :, 2]
                # img[:7, :7, 1] = local[:, :, 1]
                # img[:7, :7, 2] = local[:, :, 3]
                render_buffer.put(img)

            # Bookkeeping                
            success_total += success
            collide_total += collide
            reward_total += reward.sum().item()
            entropy_total += (entropy.cpu().numpy() * (~done)).sum()
            step_total += (~done).sum()

        # Finish episode
        stat = Stat(
            success_rate=success_total / cfg.total_agents * 100,
            reward=reward_total / cfg.total_agents,
            step=step_total / cfg.total_agents,
            collide=collide_total / cfg.total_agents,
            timeout=np.count_nonzero(~done),
            entropy=entropy_total / step_total
        )
        if cfg.output_name is not None and stat.reward > best_reward:
            print('better result with reward', stat.reward)
            write_output(cfg.output_name, output_buffer, env, step_now + 1)
            best_reward = stat.reward

        if not cfg.training:
            print('Episode %4d. Success %6.2f%%. Reward %5.2f. Collide %5.2f. Step %4.0f. Timeout %4.1f. Entropy %4.2f' % (
                i_episode, *stat))
        else:
            stats.append(stat)
            mean_stat = Stat(*tuple(sum(stat) / len(stat) for stat in zip(*stats)))

            # Generate a new stage when the model fits it
            if not cfg.fixed_reset_interval:
                stage_episode += 1
                if stat.success_rate >= cfg.master_threshold:
                    success_time += 1
                    if success_time == cfg.master_time:
                        print(
                            'Stage mastered. Episode taken: %d. New stage generated...' % stage_episode)
                        stage_episode = 0
                        env.reset_reward()
                else:
                    success_time = 0
            elif i_episode % cfg.reset_stage_every == 0:
                env.reset_reward()

            # Save checkpoints
            ctrl.update_best(mean_stat.reward, cfg.save_name)

            if i_episode % cfg.save_every == 0:
                ctrl.save(cfg.save_path)

            if not (cfg.early_stop is None) and (mean_stat.success_rate > cfg.early_stop):
                print('Early stop threshold %d: early stopping...' %
                    cfg.early_stop)
                ctrl.save('checkpoints/%s_final.pt' % cfg.save_name)
                break

            # Optimize
            loss = Loss(*ctrl.optimize(mem, cfg.batch_size))
            losses.append(loss)
            mean_losses = [sum(loss) / len(loss) for loss in zip(*losses)]

            eps = ctrl.global_episode
            writer.add_scalar('reward_agent_eps', stat.reward, eps)
            writer.add_scalar('success_rate_eps', stat.success_rate, eps)
            writer.add_scalar('step_agent_eps', stat.step, eps)
            writer.add_scalar('collision_agent_eps', stat.collide, eps)
            writer.add_scalar('entropy_agent_step', stat.entropy, eps)
            writer.add_scalar('actor_loss', loss.a_loss, eps)
            writer.add_scalar('critic_loss', loss.c_loss, eps)
            writer.add_scalar('entropy_loss', loss.e_loss, eps)

            # Logging
            if i_episode % cfg.log_every == 0:
                print('Episode %4d. Success %6.2f%%. Reward %5.2f. Collide %5.2f. Step %4.0f. Timeout %4.1f. Entropy %4.2f' % (
                    ctrl.global_episode, *mean_stat), end=' ')
                print('Losses (a: %6.3f, c: %6.3f, e: %6.3f)' % (*mean_losses, ))

    Profiler.print_all(total_step, cfg.total_episode)
    render_buffer.put(None)


def main():
    cfg = Config()
    cfg.parse_arguments()

    render_buffer = mp.Queue()

    # Start training in another process
    p = mp.Process(target=train_process, args=(cfg, render_buffer))
    p.start()
    print('Process started. Please wait...')

    if cfg.render:
        # make if possible to use ctrl+c to exit safely
        def sigint_handler(*args):
            """
            Handler for the SIGINT signal
            """
            print('Keyboard interrupted')
            QtGui.QApplication.quit()
        signal.signal(signal.SIGINT, sigint_handler)
        timer = QtCore.QTimer()
        timer.start(500)
        timer.timeout.connect(lambda: None)

        # start qt window
        app = QtGui.QApplication(sys.argv)
        viewer = Viewer(*cfg.stage_size, render_buffer)
        viewer.show()
        app.exec_()
        p.terminate()
        print('Process terminated')
    else:
        p.join()
        print('Process finished')


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
