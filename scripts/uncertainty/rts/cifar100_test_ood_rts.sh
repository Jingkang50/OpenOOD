#!/bin/bash
# sh scripts/uncertainty/rts/cifar100_test_rts_msp.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

# PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/rts_net.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/rts.yml \
--network.backbone.name resnet18_32x32 \
--num_workers 8 \
--network.checkpoint 'results/cifar100_rts_net_rts_e100_lr0.1_default/best_epoch89_acc0.7850.ckpt' \
