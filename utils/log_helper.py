#encoding: utf8
from __future__ import division

import os
import logging
import math

logs = set()

class Filter:
    def __init__(self, flag):
        self.flag = flag
    def filter(self, x): return self.flag


def get_format(logger, level):
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])

        if level == logging.INFO:
            logger.addFilter(Filter(rank == 0))
    else:
        rank = 0
    format_str = '[%(asctime)s-rk{}-%(filename)s#%(lineno)3d] %(message)s'.format(rank)
    formatter = logging.Formatter(format_str)
    return formatter


def init_log(name, level = logging.INFO):
    if (name, level) in logs: return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = get_format(logger, level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def add_file_handler(name, log_file, level = logging.INFO):
    logger = logging.getLogger(name)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(get_format(logger, level))
    logger.addHandler(fh)


init_log('global')
# init_log('global', logging.WARN)


def print_speed(i, i_time, n):
    """print_speed(index, index_time, total_iteration)"""
    logger = logging.getLogger('global')
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    logger.info('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' % (i, n, i/n*100, average_time, remaining_day, remaining_hour, remaining_min))


def main():
    for i, lvl in enumerate([logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]):
        log_name = str(lvl)
        init_log(log_name, lvl)
        logger = logging.getLogger(log_name)
        print('****cur lvl:{}'.format(lvl))
        logger.debug('debug')
        logger.info('info')
        logger.warning('warning')
        logger.error('error')
        logger.critical('critiacal')


if __name__ == '__main__':
    main()

