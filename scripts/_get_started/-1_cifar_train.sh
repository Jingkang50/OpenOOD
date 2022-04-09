#!/bin/bash
# sh scripts/_get_started/-1_cifar_train.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/networks/res18.yml \
configs/pipelines/train/baseline.yml \
--optimizer.num_epochs 100 \
--num_workers 8 \
--dataset.train.batch_size 256 \
--force_merge True
