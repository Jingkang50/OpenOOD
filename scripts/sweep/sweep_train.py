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
    parser.add_argument('--launcher',
                        default='local',
                        choices=['local', 'slurm'])
    args = parser.parse_args()

    # different command with different job schedulers
    if args.launcher == 'slurm':
        command_prefix = ("PYTHONPATH='.':$PYTHONPATH \
                          srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 \
                          --cpus-per-task=1 --ntasks-per-node=1 \
                          --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-67 ")
    else:
        command_prefix = "PYTHONPATH='.':$PYTHONPATH "

    print(f'{len(args.benchmarks)} experiments have been setup...', flush=True)
    for exp_id, benchmark in enumerate(args.benchmarks):
        print(f'Experiment #{exp_id} Starts...', flush=True)
        for sid in range(1, 6):
            print(f'5 OSR Exp, {sid} out of 5', flush=True)
            command = (f'python main.py --config \
            configs/datasets/osr_{benchmark}/{benchmark}_seed{sid}.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/networks/{network_dict[benchmark]}.yml \
            configs/pipelines/train/baseline.yml')
            os.system(command_prefix + command + ' &')
