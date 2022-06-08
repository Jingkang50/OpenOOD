# python scripts/sweep/baseline.py
import os

dataset = 'digits/mnist'
benchmarks = ['fsood', 'ood']
methods = ['msp', 'odin', 'ebo', 'mds']
mark = '0408'
model_path = './results/mnist_lenet_base_e100_lr0.1/best.ckpt'

for benchmark in benchmarks:
    for method in methods:
        exp_name = f'{benchmark}_{method}'
        command = (f"PYTHONPATH='.':$PYTHONPATH \
        srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 \
        --cpus-per-task=1 --ntasks-per-node=1 \
        --kill-on-bad-exit=1 --job-name=fsood -w SG-IDC1-10-51-2-79 \
        python main.py \
        --config configs/datasets/{dataset}.yml \
        configs/datasets/{dataset}_{benchmark}.yml \
        configs/networks/lenet.yml \
        configs/pipelines/test/test_{benchmark}.yml \
        configs/postprocessors/{method}.yml \
        --num_workers 8 \
        --network.checkpoint {model_path} \
        --output_dir ./results/mnist_{mark} \
        --mark {mark} & ")
        os.system(command)
