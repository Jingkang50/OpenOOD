#!/bin/bash
# sh scripts/uncertainty/mc_dropout/mnist_test_mc_dropout.sh

#GPU=1
#CPU=1
#node=73
#jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/dropout_net.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/dropout.yml \
--network.backbone.name lenet \
--num_workers 8 \
--network.checkpoint 'results/mnist_dropout_net_base_e100_lr0.1_default/best.ckpt' \
--mark 0
