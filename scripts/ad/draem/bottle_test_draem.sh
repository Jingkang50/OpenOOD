#!/bin/bash
# sh scripts/ad/draem/bottle_test_draem.sh

# GPU=1
# CPU=1
# node=30
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/draem/bottle.yml \
configs/networks/draem.yml \
configs/pipelines/test/test_draem.yml \
configs/preprocessors/draem_preprocessor.yml \
configs/postprocessors/msp.yml \
--evaluator.name ood
