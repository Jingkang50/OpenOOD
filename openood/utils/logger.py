import errno
import os
import os.path as osp
import sys

import yaml

import openood.utils.comm as comm


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

    Imported from
    `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`

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
    print('------------------ Config --------------------------', flush=True)
    print(config, flush=True)
    print(u'\u2500' * 70, flush=True)

    output = config.output_dir

    if config.save_output and comm.is_main_process():
        print('Output dir: {}'.format(output), flush=True)
        if osp.isdir(output):
            if config.merge_option == 'default':
                ans = input('Exp dir already exists, merge it? (y/n)')
                if ans in ['yes', 'Yes', 'YES', 'y', 'Y', 'can']:
                    save_logger(config, output)
                elif ans in ['no', 'No', 'NO', 'n', 'N']:
                    print('Quitting the process...', flush=True)
                    quit()
                else:
                    raise ValueError('Unexpected Input.')
            elif config.merge_option == 'merge':
                save_logger(config, output)
            elif config.merge_option == 'pass':
                if os.path.exists(os.path.join(config.save_output, 'ood.csv')):
                    print('Exp dir already exists, quitting the process...',
                          flush=True)
                    quit()
                else:
                    save_logger(config, output)
        else:
            save_logger(config, output)
    else:
        print('No output directory.', flush=True)

    comm.synchronize()


def save_logger(config, output):
    print('Output directory path: {}'.format(output), flush=True)
    os.makedirs(output, exist_ok=True)
    # Save config
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
