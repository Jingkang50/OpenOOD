#!/bin/bash
# sh scripts/ad/patchcore/cifar100_test_ood_patchcore.sh

GPU=1
CPU=1
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/patchcore_net.yml \
configs/pipelines/test/test_patchcore.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/patch.yml \
--num_workers 8 \
--merge_option merge
