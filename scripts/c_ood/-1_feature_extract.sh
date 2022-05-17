#!/bin/bash
# sh scripts/c_ood/-1_feature_extract.sh

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
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/feat_extract.yml \
--network.checkpoint "./results/tin_resnet18_32x32_base_e200_lr0.125/best.ckpt" \
--pipeline.extract_target train \
--force_merge True \
--num_workers 8
