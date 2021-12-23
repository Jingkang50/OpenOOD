import errno
import json
import os
import os.path as osp
import sys
from re import T

import yaml


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger:
    """Write console output to external text file.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`

    Args:
        fpath (str): directory to save logging file.

    Examples:
       >>> import sys
       >>> import os.path as osp
       >>> save_dir = 'output/experiment-1'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def setup_logger(config):
    """generate exp directory to save configs, logger, checkpoints, etc.

    Args:
        config: all configs of the experiment
    """
    output = osp.join(config.output_dir, config.exp_name)
    if osp.isdir(output):
        ans = input('Exp dir already exists, merge it? (y/n)')
        if ans in ['yes', 'Yes', 'YES', 'y', 'Y', 'can']:
            pass
        elif ans in ['no', 'No', 'NO', 'n', 'N']:
            print('Quitting the process...', flush=True)
            quit()
        else:
            raise ValueError('Unexpected Input.')
    else:
        os.makedirs(output, exist_ok=True)

    # save config
    # FIXME: saved config file is not beautified.
    config_save_path = osp.join(output, 'config.yml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config,
                  f,
                  default_flow_style=False,
                  sort_keys=False,
                  indent=2)

    # save log file
    fpath = osp.join(output, 'log.txt')
    sys.stdout = Logger(fpath)
