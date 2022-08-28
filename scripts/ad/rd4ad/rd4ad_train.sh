#!/bin/bash
# sh scripts/ad/rd4ad/rd4ad_train.sh

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/mvtec/cable.yml \
configs/pipelines/train/train_rd4ad.yml \
configs/networks/rd4ad_net.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/rd4ad.yml
