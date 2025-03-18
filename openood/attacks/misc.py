import os
import json
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

# import detectors
import timm


def convert_to_float(number):
    if number and '/' in number:
        num,den = number.split( '/' )
        number =  (float(num)/float(den))
    elif number and '.' in number:
        number = float(number)
    return number


def check_file_ending(path, ending='.json'):
    ext = os.path.splitext(path)[-1].lower()
    if ext == '':
        path = path + ending
    return path


def load_args(args, parser):
    print("load json>", args.load_json)
    filename = check_file_ending(args.load_json)
    with open(filename, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    return args


def save_args(args):
    # Optional: support for saving settings into a json file
    print("save json>", args.save_json)
    filename = check_file_ending(args.save_json)
    if filename:
        with open(filename, 'wt') as f:
            json_args = vars(args)
            del json_args['save_json']
            json.dump(json_args, f, indent=4)


def args_handling(args, parser, cfg_path):
    args.load_json = os.path.join(cfg_path, args.load_json)
    if not args.save_json == "":
        save_args(args)
    else:
        args = load_args(args, parser)
    return args


def print_args(args):
    print(''.join(f'{k}={v} \n' for k, v in vars(args).items()))


def create_dir(path):
    is_existing = os.path.exists(path)
    if not is_existing:
        os.makedirs(path)
        print("The new directory is created!", path)


def check_str_startswith(string, substring):
    start = string.split('_')[0]
    if start in substring:
        return True
    return False


def create_log_file(args, log_path):    
    create_dir(log_path)
    log = vars(args)
    return log


def save_log(args, log_dict, save_dir):
    print("save log ...", save_dir)
    print(log_dict)

    save_dir_file = os.path.join(save_dir, log_dict['timestamp_start'] + '_' + args.load_json.split('/')[-1].replace("json", "txt"))

    if hasattr(args, "tuning") and args.tuning is not None:
        save_dir_file = save_dir_file.replace(".txt", args.tuning + ".txt") 

    with open(save_dir_file , "w") as write_file:
        try:
            json.dump(log_dict, write_file, indent=4)
        finally:
            write_file.close()


def create_pth(args, ws_path, filename, dataset, join=True):
    save_dir = os.path.join(ws_path, dataset)
    if ws_path == ws_extract_path:
        save_dir = os.path.join(ws_path, dataset, args.extract)
    
    if join:
        to_save = os.path.join(save_dir, filename)
        print("save> ",  to_save)
        return to_save

    return save_dir, filename


def save_to_pt(args, ws_path, payload, filename):
    savedir, filename = create_pth(args, ws_path, filename, args.dataset, join=False)
    create_dir(savedir)
    pth_filename = os.path.join(savedir, filename)
    print("save to> ", pth_filename)
    
    torch.save(payload,  pth_filename)


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_preprocessing(args):

    if args.dataset == 'cifar10':
        preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.247,  0.2435, 0.2616], axis=-3)
    elif args.dataset == 'cifar100':
        preprocessing = dict(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761], axis=-3)
    elif args.dataset == 'imagenet':
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

    return preprocessing

