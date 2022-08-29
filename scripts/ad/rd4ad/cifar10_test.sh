#!/bin/bash
# sh scripts/ad/rd4ad/cifar10_test.sh

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
#--config configs/datasets/mvtec/cable.yml \
python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/rd4ad_net.yml \
configs/pipelines/test/test_rd4ad.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/rd4ad.yml \
