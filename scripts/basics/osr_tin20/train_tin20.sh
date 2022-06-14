#!/bin/bash
# sh scripts/basics/osr_tin20/train_tin20.sh
# python -m pdb -c continue main.py \

GPU=1
CPU=1
node=75
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/osr_tin20/tin20_seed1.yml \
configs/networks/resnet18_64x64.yml \
configs/preprocessors/base_preprocessor.yml \
configs/pipelines/train/baseline.yml \
--network.pretrained False \
--dataset.image_size 64 \
--optimizer.num_epochs 100 \
--num_workers 4 \
--mark 5 &
