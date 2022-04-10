#!/bin/bash
# sh scripts/_get_started/-1_imagenet_train.sh

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
configs/networks/res50.yml \
configs/pipelines/test/test_acc.yml \
--num_workers 8 \
--dataset.train.batch_size 2 \
--save_output False
