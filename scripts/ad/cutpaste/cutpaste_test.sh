#!/bin/bash
# sh scripts/a_anomaly/2_cutpaste_test.sh

GPU=1
CPU=1
node=68
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/draem/bottle.yml \
configs/networks/cutpaste.yml \
configs/pipelines/test/test_cutpaste.yml \
configs/postprocessors/cutpaste.yml \
configs/preprocessors/cutpaste_preprocessor.yml \
--network.checkpoint "results/bottle_projectionNet_cutpaste_e100_lr0.03/best_epoch35_auroc93.08219178082192.ckpt" \
--evaluator.name ad
