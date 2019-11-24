import sys
import signal

import torch
import torch.multiprocessing as mp
import numpy as np

from pyqtgraph.Qt import QtGui, QtCore

from config import Config
from controller import Controller
from environment.env import Environment
from viewer import Viewer
from utils import Profiler
from logger import Stat


def train_process(cfg: Config, render_buffer: mp.Queue):
    """
    Main train loop
    """
    # Global environment
    cfg.global_deterministic()
    device = torch.device(cfg.device)
    Profiler.enable_profiling(cfg.profiling)

    # to_print = []
    # for num_agents in [128]:
    # cfg.total_agents = num_agents
    Profiler.reset()

    # Init
    env = Environment(cfg, device)
    ctrl = Controller(cfg, env, device)

    # stats = []

    # Run training or inference
    for i_episode in range(1, cfg.total_episode + 1):
        with Profiler('episode'):
            stat = ctrl.run_episode(i_episode, render_buffer)
            # stats.append(stat)

    # Finishing
    # mean_stat = Stat(*tuple(sum(stat) / len(stat)for stat in zip(*stats)))
    t_step, t_eps = Profiler.print_all(ctrl.total_step, cfg.total_episode)

    # to_print.append((num_agents, mean_stat, t_step, t_eps))

    # for log in to_print:
    #     print('Agents %d. Success %6.2f%%. Reward %5.2f. Collide %5.2f. Step %4.0f. Timeout %4.1f. Entropy %4.2f' % (
    #             log[0], *log[1]))
    #     print('Step %d. Episode %d' % (log[2], log[3]))

    render_buffer.put(None)


def main():
    cfg = Config()
    cfg.parse_arguments()

    render_buffer = mp.Queue()

    # Start training in another process
    p = mp.Process(target=train_process, args=(cfg, render_buffer))
    p.start()
    print('Simulation started in new process')

    if cfg.render:
        # Enable ctrl+c to exit safely
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

        # Start qt window for viewing
        print('Starting a rendering window...')
        app = QtGui.QApplication(sys.argv)
        viewer = Viewer(*cfg.stage_size, render_buffer)
        viewer.show()
        app.exec_()
        p.terminate()
        print('Process terminated')
    else:
        # Without rendering, just start the process
        p.join()
        print('Process finished')


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
