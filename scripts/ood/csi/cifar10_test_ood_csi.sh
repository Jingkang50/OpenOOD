#!/bin/bash
# sh scripts/ood/csi/cifar10_test_ood_csi.sh

GPU=1
CPU=1
node=36
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/csi_net.yml \
    configs/pipelines/test/test_ood.yml \
    configs/postprocessors/msp.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint 'results/cifar10_csi_net_csi_step2_e100_lr0.1/s0/best.ckpt' \
    --merge_option merge

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_csi_net_csi_step2_e100_lr0.1 \
    --postprocessor msp \
    --save-score --save-csv
