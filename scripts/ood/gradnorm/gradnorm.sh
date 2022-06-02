#!/bin/bash
# sh scripts/c_ood/0_mnist_test_fsood_msp.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/imagenet/imagenet.yml \
configs/datasets/imagenet/imagenet_ood.yml \
configs/networks/bit.yml \
configs/pipelines/test/test_ood.yml \
configs/postprocessors/gradnorm.yml \
--num_workers 4 \
--network.checkpoint ./bit_pretrained_models/BiT-S-R101x1.npz \
--mark 0
