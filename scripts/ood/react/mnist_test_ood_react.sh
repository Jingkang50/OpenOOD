#!/bin/bash
# sh scripts/ood/react/mnist_test_ood_react.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/react_net.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/react.yml \
--network.pretrained False \
--network.backbone.name lenet \
--network.backbone.pretrained True \
--network.backbone.checkpoint 'results/checkpoints/mnist_lenet_acc99.30.ckpt' \
--num_workers 8 \
--mark 0
