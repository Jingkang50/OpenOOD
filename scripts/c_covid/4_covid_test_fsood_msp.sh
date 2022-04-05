#!/bin/bash
# sh scripts/c_covid/4_covid_test_fsood_msp.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/covid/covid.yml \
configs/datasets/covid/covid_fsood.yml \
configs/networks/resnet18_224x224.yml \
configs/pipelines/test/test_fsood.yml \
configs/postprocessors/msp.yml \
--network.checkpoint ./results/covid_resnet18_224x224_base_e50_lr0.001/last.ckpt \
--num_workers 4 \
--mark 0405
