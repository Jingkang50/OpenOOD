import argparse
import os
import re

import yaml


def setup_config(config_process_order=('merge', 'parse_args', 'parse_refs')):
    """Parsing configuration files and command line augments.

    This method reads the command line to
        1. extract and stack YAML config files,
        2. collect modification in command line arguments,
    so that the finalized configuration file is generated.

    Note:
        The default arguments allow the following equivalent code:
            config = merge_configs(*config)
                --> merge multiple YAML config files
            config.parse_args(unknown_args)
                --> use command line arguments to overwrite default settings
            config.parse_refs()
                --> replace '@{xxx.yyy}'-like values with referenced values
        It is recommended to merge before parse_args so that the latter configs
        can re-use references in the previous configs.
        For example, if
            config1.key1 = jkyang
            config1.key2 = '@{key1}'
            config2.key1 = yzang
            config3 = merge_configs(config1, config2)
            config3.parse_refs()
        then
            config3.key2 will be yzang rather than jkyang

    Return:
        An object of <class 'openood.utils.config.Config'>.
        Can be understanded as a dictionary.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', nargs='+', required=True)
    opt, unknown_args = parser.parse_known_args()
    config = [Config(path) for path in opt.config]

    for process in config_process_order:
        if process == 'merge':
            config = merge_configs(*config)
        elif process == 'parse_args':
            if isinstance(config, Config):
                config.parse_args(unknown_args)
            else:
                for cfg in config:
                    cfg.parse_args(unknown_args)
        elif process == 'parse_refs':
            if isinstance(config, Config):
                config.parse_refs()
            else:
                for cfg in config:
                    cfg.parse_refs()
        else:
            raise ValueError('unknown config process name: {}'.format(process))

    # manually modify 'output_dir'
    config.output_dir = os.path.join(config.output_dir, config.exp_name)

    return config


def parse_config(config):
    config_process_order = ('merge', 'parse_refs')
    for process in config_process_order:
        if process == 'merge':
            config = merge_configs(*config)
        elif process == 'parse_refs':
            if isinstance(config, Config):
                config.parse_refs()
            else:
                for cfg in config:
                    cfg.parse_refs()
        else:
            raise ValueError('unknown config process name: {}'.format(process))
    # manually modify 'output_dir'
    config.output_dir = os.path.join(config.output_dir, config.exp_name)

    return config


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__()
        for arg in args:
            if arg == ' ':
                continue  # hard code remove white space in config file list
            if isinstance(arg, str):
                if arg.endswith('.yml'):
                    with open(arg, 'r') as f:
                        raw_dict = yaml.safe_load(f)
                else:
                    raise Exception('unknown file format %s' % arg)
                init_assign(self, raw_dict)
            elif isinstance(arg, dict):
                init_assign(self, arg)
            else:
                raise TypeError('arg should be an instance of <str> or <dict>')
        if kwargs:
            init_assign(self, kwargs)

    def __call__(self, *args, **kwargs):
        return Config(self, *args, **kwargs)

    def __repr__(self, indent=4, prefix=''):
        r = []
        for key, value in sorted(self.items()):
            if isinstance(value, Config):
                r.append('{}{}:'.format(prefix, key))
                r.append(value.__repr__(indent, prefix + ' ' * indent))
            else:
                r.append('{}{}: {}'.format(prefix, key, value))
        return '\n'.join(r)

    def __setstate__(self, state):
        init_assign(self, state)

    def __getstate__(self):
        d = dict()
        for key, value in self.items():
            if type(value) is Config:
                value = value.__getstate__()
            d[key] = value
        return d

    # access by '.' -> access by '[]'
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    # access by '[]'
    def __getitem__(self, key):
        sub_cfg, sub_key = consume_dots(self, key, create_default=False)
        return dict.__getitem__(sub_cfg, sub_key)

    def __setitem__(self, key, value):
        sub_cfg, sub_key = consume_dots(self, key, create_default=True)
        if sub_cfg.__contains__(sub_key) and value == '_DELETE_CONFIG_':
            dict.__delitem__(sub_cfg, sub_key)
        else:
            dict.__setitem__(sub_cfg, sub_key, value)

    def __delitem__(self, key):
        sub_cfg, sub_key = consume_dots(self, key, create_default=False)
        dict.__delitem__(sub_cfg, sub_key)

    # access by 'in'
    def __contains__(self, key):
        try:
            sub_cfg, sub_key = consume_dots(self, key, create_default=False)
        except KeyError:
            return False
        return dict.__contains__(sub_cfg, sub_key)

    # traverse keys / values/ items
    def all_keys(self, only_leaf=True):
        for key in traverse_dfs(self,
                                'key',
                                continue_type=Config,
                                only_leaf=only_leaf):
            yield key

    def all_values(self, only_leaf=True):
        for value in traverse_dfs(self,
                                  'value',
                                  continue_type=Config,
                                  only_leaf=only_leaf):
            yield value

    def all_items(self, only_leaf=True):
        for key, value in traverse_dfs(self,
                                       'item',
                                       continue_type=Config,
                                       only_leaf=only_leaf):
            yield key, value

    # for command line arguments
    def parse_args(self, cmd_args=None, strict=True):
        unknown_args = []
        if cmd_args is None:
            import sys
            cmd_args = sys.argv[1:]
        index = 0
        while index < len(cmd_args):
            arg = cmd_args[index]
            err_msg = 'invalid command line argument pattern: %s' % arg
            assert arg.startswith('--'), err_msg
            assert len(arg) > 2, err_msg
            assert arg[2] != '-', err_msg

            arg = arg[2:]
            if '=' in arg:
                key, full_value_str = arg.split('=')
                index += 1
            else:
                assert len(
                    cmd_args) > index + 1, 'incomplete command line arguments'
                key = arg
                full_value_str = cmd_args[index + 1]
                index += 2
            if ':' in full_value_str:
                value_str, value_type_str = full_value_str.split(':')
                value_type = eval(value_type_str)
            else:
                value_str = full_value_str
                value_type = None

            if key not in self:
                if strict:
                    raise KeyError(key)
                else:
                    unknown_args.extend(['--' + key, full_value_str])
                    continue

            if value_type is None:
                value_type = type(self[key])

            if value_type is bool:
                self[key] = {
                    'true': True,
                    'True': True,
                    '1': True,
                    'false': False,
                    'False': False,
                    '0': False,
                }[value_str]
            else:
                self[key] = value_type(value_str)

        return unknown_args

    # for key reference
    def parse_refs(self, subconf=None, stack_depth=1, max_stack_depth=10):
        if stack_depth > max_stack_depth:
            raise Exception(
                ('Recursively calling `parse_refs` too many times'
                 'with stack depth > {}. '
                 'A circular reference may exists in your config.\n'
                 'If deeper calling stack is really needed,'
                 'please call `parse_refs` with extra argument like: '
                 '`parse_refs(max_stack_depth=9999)`').format(max_stack_depth))
        if subconf is None:
            subconf = self
        for key in subconf.keys():
            value = subconf[key]
            if type(value) is str and '@' in value:
                if value.count('@') == 1 and value.startswith(
                        '@{') and value.endswith('}'):
                    # pure reference
                    ref_key = value[2:-1]
                    ref_value = self[ref_key]
                    subconf[key] = ref_value
                else:
                    # compositional references
                    ref_key_list = re.findall("'@{(.+?)}'", value)
                    ref_key_list = list(set(ref_key_list))
                    ref_value_list = [
                        self[ref_key] for ref_key in ref_key_list
                    ]
                    origin_ref_key_list = [
                        "'@{" + ref_key + "}'" for ref_key in ref_key_list
                    ]
                    for origin_ref_key, ref_value in zip(
                            origin_ref_key_list, ref_value_list):
                        value = value.replace(origin_ref_key, str(ref_value))
                    subconf[key] = value
        for key in subconf.keys():
            value = subconf[key]
            if type(value) is Config:
                self.parse_refs(value, stack_depth + 1)


def merge_configs(*configs):
    final_config = Config()
    for i in range(len(configs)):
        config = configs[i]
        if not isinstance(config, Config):
            raise TypeError(
                'config.merge_configs expect `Config` type inputs, '
                'but got `{}`.\n'
                'Correct usage: merge_configs(config1, config2, ...)\n'
                'Incorrect usage: merge_configs([configs1, configs2, ...])'.
                format(type(config)))
        final_config = final_config(dict(config.all_items()))
    return final_config


def consume_dots(config, key, create_default):
    sub_keys = key.split('.', 1)
    sub_key = sub_keys[0]

    if sub_key in Config.__dict__:
        raise KeyError(
            '"{}" is a preserved API name, '
            'which should not be used as normal dictionary key'.format(
                sub_key))

    if not dict.__contains__(config, sub_key) and len(sub_keys) == 2:
        if create_default:
            dict.__setitem__(config, sub_key, Config())
        else:
            raise KeyError(key)

    if len(sub_keys) == 1:
        return config, sub_key
    else:
        sub_config = dict.__getitem__(config, sub_key)
        if type(sub_config) != Config:
            if create_default:
                sub_config = Config()
                dict.__setitem__(config, sub_key, sub_config)
            else:
                raise KeyError(key)
        return consume_dots(sub_config, sub_keys[1], create_default)


def traverse_dfs(root, mode, continue_type, only_leaf, key_prefix=''):
    for key, value in root.items():
        full_key = '.'.join([key_prefix, key]).strip('.')
        child_kvs = []
        if type(value) == continue_type:
            for kv in traverse_dfs(value, mode, continue_type, only_leaf,
                                   full_key):
                child_kvs.append(kv)
        # equivalent:
        # if not (len(child_kvs) > 0 and
        # type(value) == continue_type and
        # only_leaf)
        if len(child_kvs
               ) == 0 or type(value) != continue_type or not only_leaf:
            yield {
                'key': full_key,
                'value': value,
                'item': (full_key, value)
            }[mode]
        for kv in child_kvs:
            yield kv


def init_assign(config, d):
    for full_key, value in traverse_dfs(d,
                                        'item',
                                        continue_type=dict,
                                        only_leaf=True):
        sub_cfg, sub_key = consume_dots(config, full_key, create_default=True)
        sub_cfg[sub_key] = value
