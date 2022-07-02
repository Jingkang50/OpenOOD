#!/bin/bash
# sh scripts/ad/patchcore/osr_cifar50_test_ood_patchcore.sh

GPU=1
CPU=1
node=30
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/osr_cifar50/cifar50_seed1.yml \
configs/datasets/osr_cifar50/cifar50_seed1_ood.yml \
configs/networks/patchcore_net.yml \
configs/pipelines/test/test_patchcore.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/patch.yml \
--network.backbone.name resnet18_32x32 \
--network.backbone.checkpoint 'results/checkpoints/osr/cifar50_seed1_acc80.24.ckpt' \
--num_workers 8 \
--merge_option merge &
