#!/bin/bash
# sh scripts/c_ood/9_cifar_test_fsood_sem.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/datasets/objects/cifar10_fsood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_fsood.yml \
configs/postprocessors/gmm_iter/cifar_gmm_0.yml \
--network.checkpoint ./results/cifar10_resnet18_32x32_sae_e10_lr0.01/best_epoch8_acc0.943.ckpt \
--num_workers 4 \
--mark sem_0331
