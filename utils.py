import sys
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def to_pixel(coord: np.ndarray):
    """
    Convert coordinates to pixel index
    """
    return coord.astype(int)


def same_pixel(p1: np.ndarray, p2: np.ndarray):
    """
    Check if two points are in the same pixel
    """
    return np.array_equal(p1.astype(int), p2.astype(int))


def clip_length(vector, maximum, axis=1):
    """
    Clip the size of a vector along an axis
    """
    lens = np.linalg.norm(vector, axis=axis)
    overflow = lens > maximum
    if np.any(overflow):
        vector[overflow] = vector[overflow] / lens[overflow, None] * maximum
    return vector


def out_of_bound(p, size):
    return p[0] < 0 or p[0] >= size[0] or p[1] < [0] or p[1] >= size[1]


def timestamp_ms():
    return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)


def get_quad(y, x, h, w):
    return np.array([
        [y, x],
        [y + h, x],
        [y + h, x + w],
        [y, x + w]
    ])


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        # init.constant_(m.weight, 1)
        # init.constant_(m.bias, 0)
        init.normal_(m.weight, mean=1, std=0.02)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        init.normal_(m.bias)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param)
            else:
                init.normal_(param)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param)
            else:
                init.normal_(param)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param)
            else:
                init.normal_(param)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param)
            else:
                init.normal_(param)


class Agents(object):
    """
    Agent batch

    Attributes:
        p:  position, array of shape (N, 2)
        v:  velocity, array of shape (N, 2)
        a:  acceleration, array of shape (N, 2)
    """

    def __init__(self, p, v, a):
        self.p = p
        self.v = v
        self.a = a


class PolygonGenerator():
    """
    An easy-to-use polygon brush tool
    """

    def __init__(self):
        self.points = []

    def begin_drawing(self, y, x):
        self.points.append(np.array([y, x]))
        return self

    def next(self, y, x):
        self.points.append(self.points[-1] + np.array([y, x]))
        return self

    def get_polygon(self):
        return np.array(self.points)
        

class Profiler():
    """
    A global profiler

    * Call enable_profiling to enable this class
    * Call record_begin('your_key') and record_end('your_key') before and after your code block
    """
    run_time = {}
    last_time = {}
    profiling = False

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        Profiler.record_begin(self.name)

    def __exit__(self, type, value, traceback):
        Profiler.record_end(self.name)

    @staticmethod
    def enable_profiling(profiling):
        Profiler.profiling = profiling

    @staticmethod
    def record_begin(name):
        if Profiler.profiling:
            Profiler.last_time[name] = timestamp_ms()

    @staticmethod
    def record_end(name):
        if Profiler.profiling:
            now = timestamp_ms()
            took = now - Profiler.last_time[name]
            if name in Profiler.run_time:
                Profiler.run_time[name] = Profiler.run_time[name] + took
            else:
                Profiler.run_time[name] = took

    @staticmethod
    def print_all(total_step, total_episode):
        if Profiler.profiling:
            total_time = 0
            for key in Profiler.run_time:
                total_time += Profiler.run_time[key]
            print('--------------- Profiler Statistics ---------------')
            for key, value in Profiler.run_time.items():
                print('%s: total %d, %f%%' %
                      (key, value, value / total_time * 100))
            print('time per step: %d' % (total_time / total_step))
            print('time per episode: %d' % (total_time / total_episode))
            print('----------------------- end -----------------------')


class Logger(object):
    """
    A logger that output to both terminal and file

    https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting/14906787
    """

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
