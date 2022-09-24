#!/bin/bash
# sh scripts/basics/osr_mnist6/train_mnist6.sh

GPU=1
CPU=1
node=78
jobname=openood

if [ $USER == "jkyang" ]; then
    PYTHONPATH='.':$PYTHONPATH \
    srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
    --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
    --kill-on-bad-exit=1 --job-name=${jobname} \
    python main.py \
    --config configs/datasets/osr_mnist6/mnist6_seed2.yml \
    configs/networks/lenet.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/baseline.yml \
    --network.pretrained False
else
    PYTHONPATH='.':$PYTHONPATH \
    python main.py \
    --config configs/datasets/osr_mnist6/mnist6_seed1.yml \
    configs/networks/lenet.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/baseline.yml \
    --network.pretrained False \
    --dataset.image_size 28 \
    --optimizer.num_epochs 100 \
    --num_workers 4
fi

cp ./results/cifar6_seed5_resnet18_32x32_base_e100_lr0.1_default/best.ckpt ./results/checkpoints/osr/cifar6_seed5.ckpt
