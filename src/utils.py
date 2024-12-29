import os
import random
import yaml
import torch
import numpy as np
from datetime import datetime


def set_all_seeds(seed):
    """
    Ensures reproducible behaviour by resetting all seeds with the seed given by `seed`.
    Moreover, additional parameters are set to ensure deterministic behaviour.

    Reference:
    [1] https://pytorch.org/docs/stable/notes/randomness.html, Accessed: 2021-07-19

    Args:
        seed: The desired seed to be set
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_logging_schema(level="INFO", file_name=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "info.log")):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'loggers': {
            '': {  # root logger
                'level': 'DEBUG',
                'handlers': ['stream_handler', 
                             'file_handler', 
                            #  'info_rotating_file_handler', 
                            #  'error_file_handler', 
                            #  'critical_mail_handler'
                             ],
            },
            # 'my.package': { 
            #     'level': 'WARNING',
            #     'propagate': False,
            #     'handlers': ['info_rotating_file_handler', 'error_file_handler' ],
            # },
        },
        'handlers': {
            'stream_handler': {
                'level': f'{level}',
                'formatter': 'debug',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },
            'file_handler': {
                'level': 'INFO',
                'formatter': 'info',
                'class': 'logging.FileHandler',
                'filename': f'{file_name}',
                # 'mode': 'a',
                # 'maxBytes': 1048576,
                # 'backupCount': 10
            },
            # 'info_rotating_file_handler': {
            #     'level': 'INFO',
            #     'formatter': 'info',
            #     'class': 'logging.handlers.RotatingFileHandler',
            #     'filename': 'info.log',
            #     'mode': 'a',
            #     'maxBytes': 1048576,
            #     'backupCount': 10
            # },
            # 'error_file_handler': {
            #     'level': 'WARNING',
            #     'formatter': 'error',
            #     'class': 'logging.FileHandler',
            #     'filename': 'error.log',
            #     'mode': 'a',
            # },
            # 'critical_mail_handler': {
            #     'level': 'CRITICAL',
            #     'formatter': 'error',
            #     'class': 'logging.handlers.SMTPHandler',
            #     'mailhost' : 'localhost',
            #     'fromaddr': 'monitoring@domain.com',
            #     'toaddrs': ['dev@domain.com', 'qa@domain.com'],
            #     'subject': 'Critical error with application name'
            # }
        },
        'formatters': {
            'info': {
                'format': '[%(asctime)s - %(levelname)s - %(module)s] %(message)s'
            },
            'debug': {
                'format': '[%(asctime)s - %(levelname)s - pid-%(process)s - %(module)s - line-%(lineno)s] %(message)s'
            },
        },
    }


def add_params_to_yaml(yml_path, args):
    with open(yml_path, "w") as fp:
        yaml.dump(args.__dict__, indent=4, stream=fp)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
