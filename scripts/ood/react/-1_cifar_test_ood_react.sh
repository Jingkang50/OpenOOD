#!/bin/bash
# sh scripts/c_ood/-1_cifar_test_ood_react.sh

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
configs/datasets/objects/cifar10_ood.yml \
configs/networks/reactnet.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/react.yml \
--num_workers 8 \
--dataset.val.batch_size 32 \
--network.pretrained False \
--network.backbone.name resnet18_32x32 \
--network.backbone.pretrained True \
--network.backbone.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1/best.ckpt' \
--save_output True
