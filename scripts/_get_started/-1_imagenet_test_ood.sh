#!/bin/bash
# sh scripts/_get_started/-1_imagenet_test_ood.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/imagenet.yml \
configs/datasets/digits/mnist_ood.yml \
configs/networks/res50.yml \
configs/pipelines/test/test_ood.yml \
configs/postprocessors/msp.yml \
--num_workers 20 \
--dataset.train.batch_size 1024 \
--dataset.test.batch_size 512 \
--network.pretrained False \
--save_output True
