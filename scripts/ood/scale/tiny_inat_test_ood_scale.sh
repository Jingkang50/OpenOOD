#!/bin/bash
# sh scripts/ood/scale/cifar10_test_ood_scale.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/scale.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 1

