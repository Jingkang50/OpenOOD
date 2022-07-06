import argparse
import csv
import os

import numpy as np
from write_metrics import make_args_list, write_metric, write_total

# dictionary with keywords from benchmarks
network_dict = {
    'mnist': 'lenet',
    'mnist6': 'lenet',
    'cifar10': 'resnet18_32x32',
    'cifar6': 'resnet18_32x32',
    'cifar100': 'resnet18_32x32',
    'cifar50': 'resnet18_32x32',
    'imagenet': 'resnet50',
    'tin20': 'resnet18_64x64'
}

checkpoint_dict = {
    'mnist': './results/checkpoints/mnist_lenet_acc98.50.ckpt',
    'cifar10': './results/checkpoints/cifar10_res18_acc95.24.ckpt',
    'cifar100': './results/checkpoints/cifar100_res18_acc77.10.ckpt',
    'imagenet': './results/checkpoints/imagenet_res50_acc76.17.pth',
    'mnist6': './results/checkpoints/osr/mnist6',
    'cifar6': './results/checkpoints/osr/cifar6',
    'cifar50': './results/checkpoints/osr/cifar50',
    'tin20': './results/checkpoints/osr/tin20',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('--benchmarks',
                        nargs='+',
                        default=['mnist', 'cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--methods', nargs='+', default=['msp'])
    parser.add_argument('--metrics', nargs='+', default=['acc'])
    parser.add_argument('--metric2save', nargs='+', default=['auroc'])
    parser.add_argument('--update_form_only', action='store_true')
    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--launcher',
                        default='local',
                        choices=['local', 'slurm'])
    parser.add_argument('--merge-option', default='default')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # different command with different job schedulers
    if args.launcher == 'slurm':
        command_prefix = ("PYTHONPATH='.':$PYTHONPATH \
                          srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 \
                          --cpus-per-task=1 --ntasks-per-node=1 \
                          --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-79 ")
    else:
        command_prefix = "PYTHONPATH='.':$PYTHONPATH "

    # TODO: dynamic benchmark dict
    benchmark_dict = {
        'ood': ['cifar10', 'cifar100'],
        'osr': ['cifar6', 'cifar50', 'mnist6', 'tin20'],
        'acc': args.benchmarks
    }

    args_list = make_args_list(args.benchmarks, args.methods, args.metrics,
                               benchmark_dict)
    print(f'{len(args_list)} experiments have been setup...', flush=True)

    if not args.update_form_only:
        for exp_id, [benchmark, method, metric] in enumerate(args_list):
            print(f'Experiment #{exp_id+1} Starts...', flush=True)
            print(f'Config: {benchmark}, {method}, {metric}', flush=True)
            if metric in ['ood', 'fsood']:
                command = (f'python main.py --config \
                configs/datasets/{benchmark}/{benchmark}.yml \
                configs/datasets/{benchmark}/{benchmark}_{metric}.yml \
                configs/preprocessors/base_preprocessor.yml \
                configs/networks/{network_dict[benchmark]}.yml \
                configs/pipelines/test/test_{metric}.yml \
                configs/postprocessors/{method}.yml \
                --network.checkpoint {checkpoint_dict[benchmark]} \
                --merge_option {args.merge_option} \
                --output_dir {args.output_dir}')
                os.system(command_prefix + command)
            elif metric == 'osr':
                for sid in range(1, 6):
                    print(f'5 OSR Exp, {sid} out of 5', flush=True)
                    command = (f'python main.py --config \
                    configs/datasets/osr_{benchmark}/{benchmark}_seed{sid}.yml \
                    configs/datasets/osr_{benchmark}/{benchmark}_seed{sid}_osr.yml \
                    configs/preprocessors/base_preprocessor.yml \
                    configs/networks/{network_dict[benchmark]}.yml \
                    configs/pipelines/test/test_osr.yml \
                    configs/postprocessors/{method}.yml \
                    --network.checkpoint {checkpoint_dict[benchmark]}_seed{sid}.ckpt \
                    --output_dir {args.output_dir} \
                    --merge_option {args.merge_option}')
                    os.system(command_prefix + command)
            elif metric in ['acc', 'ece']:
                command = (f'python main.py --config \
                configs/datasets/{benchmark}/{benchmark}.yml \
                configs/preprocessors/base_preprocessor.yml \
                configs/networks/{network_dict[benchmark]}.yml \
                configs/pipelines/test/test_{metric}.yml \
                configs/postprocessors/{method}.yml \
                --network.checkpoint {checkpoint_dict[benchmark]} \
                --output_dir {args.output_dir} \
                --merge_option {args.merge_option}')
                os.system(command_prefix + command)
            else:
                raise ValueError('Unexpected Metric...')

    folder_list = os.listdir(args.output_dir)
    # TODO: do not hard code -8
    save_line_dict = {'ood': -8, 'osr': -1, 'acc': -1}
    # TODO: extend according to config
    args.benchmarks.extend([
        'tin', 'nearood', 'mnist', 'svhn', 'texture', 'place365', 'places365',
        'farood'
    ])

    # TODO: try to find farood and near ood in another way, user can se t what to save by changing ood's list
    main_content_extract_dict = {'ood': ['nearood', 'farood'], 'osr': [-1]}
    write_metric(args, folder_list, save_line_dict, benchmark_dict)
    write_total(args, folder_list, save_line_dict, benchmark_dict,
                main_content_extract_dict)
