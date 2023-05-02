#!/bin/bash
# sh scripts/ood/conf_branch/cifar10_train_conf_branch.sh

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/conf_branch.yml \
    configs/pipelines/train/train_conf_branch.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.backbone.name resnet18_32x32 \
    --seed ${SEED}
