import argparse
import os

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

method_dict = {
    'msp':
    None,
    'odin': [
        '--postprocessor.postprocessor_args.temperature 1',
        '--postprocessor.postprocessor_args.temperature 100',
        '--postprocessor.postprocessor_args.temperature 1000'
    ],
    'mds':
    None,
    'gram':
    None,
}


def make_args_list(benchmarks, methods, metrics):
    args_list = []
    for benchmark in benchmarks:
        for method in methods:
            for metric in metrics:
                args_list.append([benchmark, method, metric])
    return args_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('--benchmarks',
                        nargs='+',
                        default=['mnist', 'cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--methods', nargs='+', default=['msp'])
    parser.add_argument('--metrics', nargs='+', default=['acc'])
    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--launcher',
                        default='local',
                        choices=['local', 'slurm'])
    args = parser.parse_args()

    # different command with different job schedulers
    if args.launcher == 'slurm':
        command_prefix = ("PYTHONPATH='.':$PYTHONPATH \
                          srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 \
                          --cpus-per-task=1 --ntasks-per-node=1 \
                          --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-79 ")
    else:
        command_prefix = "PYTHONPATH='.':$PYTHONPATH "

    args_list = make_args_list(args.benchmarks, args.methods, args.metrics)
    print(f'{len(args_list)} experiments have been setup...', flush=True)
    for exp_id, [benchmark, method, metric] in enumerate(args_list):
        print(f'Experiment #{exp_id} Starts...', flush=True)
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
            --output_dir {args.output_dir}')
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
                --output_dir {args.output_dir}')
                os.system(command_prefix + command)
        elif metric in ['acc', 'ece']:
            command = (f'python main.py --config \
            configs/datasets/{benchmark}/{benchmark}.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/networks/{network_dict[benchmark]}.yml \
            configs/pipelines/test/test_{metric}.yml \
            configs/postprocessors/{method}.yml \
            --network.checkpoint {checkpoint_dict[benchmark]} \
            --output_dir {args.output_dir}')
            os.system(command_prefix + command)
        else:
            raise ValueError('Unexpected Metric...')
