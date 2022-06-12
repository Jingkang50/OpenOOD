#!/bin/bash
# sh scripts/ood/knn/mnist_test_osr_knn.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/osr_mnist6/mnist6_seed1.yml \
configs/datasets/osr_mnist6/mnist6_seed1_ood.yml \
configs/networks/lenet.yml \
configs/pipelines/test/test_osr.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/knn.yml \
--num_workers 8 \
--network.checkpoint 'results/checkpoints/osr/mnist6_seed1.ckpt' \
--mark 0
