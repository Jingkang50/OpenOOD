#!/bin/bash
# sh scripts/ood/vos/cifar100_test_vos.sh
#in cifar100_resnet18_32x32_test_ood_ood_ebo_0 directiry
PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/ebo.yml \
--num_workers 8 \
--network.pretrained True \
--network.checkpoint 'results/cifar100_resnet18_32x32_vos/best.ckpt' \
--mark 0
