#!/bin/bash
# sh scripts/uncertainty/pixmix/cifar10_test_ood_msp.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --num_workers 8 \
    --network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_pixmix/s0/best.ckpt' \
    --mark pixmix

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_pixmix \
   --postprocessor msp \
   --save-score --save-csv
