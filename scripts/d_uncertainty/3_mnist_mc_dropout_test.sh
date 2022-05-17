#!/bin/bash
# sh scripts/d_uncertainty/3_mnist_dropout_test.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/digits/mnist.yml \
configs/datasets/digits/mnist_fsood.yml \
configs/networks/lenet.yml \
configs/pipelines/test/test_fsood.yml \
configs/postprocessors/dropout.yml \
configs/preprocessors/base_preprocessor.yml \
--num_workers 4 \
--mark 0 \
--exp_name mnist_lenet_e50_lr0.1_test_fsood_mc_dropout \
--network.checkpoint ./results/lenet_dropout_pretrained/mnist_lenet_dropout_e10_lr0.1/best.ckpt
