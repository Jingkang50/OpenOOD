#!/bin/bash
# sh scripts/a_anomaly/-1_DRAEM_test.sh

# GPU=1
# CPU=1
# node=30
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/DRAEM/bottle.yml \
configs/networks/DRAEM.yml \
configs/pipelines/test/test_DRAEM.yml \
configs/preprocessors/draem_preprocessor.yml
