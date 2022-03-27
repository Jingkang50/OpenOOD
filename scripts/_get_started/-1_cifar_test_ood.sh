#!/bin/bash
# sh scripts/_get_started/-1_cifar_test_ood.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/datasets/digits/mnist_ood.yml \
configs/networks/res18.yml \
configs/pipelines/test/test_ood.yml \
configs/postprocessors/react.yml \
--num_workers 8 \
--dataset.train.batch_size 256 \
--force_merge False \
--network.pretrained True \
--network.checkpoint ./results/cifar10_res18_base_e100_lr0.1/best.ckpt
