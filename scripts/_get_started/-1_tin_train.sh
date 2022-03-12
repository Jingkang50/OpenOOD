#!/bin/bash
# sh scripts/_get_started/-1_tin_train.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/objects/tin.yml \
configs/networks/res18.yml \
configs/pipelines/train/baseline.yml \
--dataset.image_size 64 \
--optimizer.num_epochs 200 \
--num_workers 8 \
--dataset.train.batch_size 256 \
--optimizer.lr 0.125 \
--optimizer.weight_decay 0.0005 \
--force_merge True
