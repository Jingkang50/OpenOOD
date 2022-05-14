#!/bin/bash
# sh scripts/c_ood/-1_cifar_test_ood_csi.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/datasets/objects/cifar10_ood.yml \
configs/networks/csinet.yml \
configs/pipelines/test/test_ood.yml \
configs/postprocessors/msp.yml \
--num_workers 8 \
--dataset.val.batch_size 32 \
--dataset.train.batch_size 32 \
--dataset.test.batch_size 32 \
--network.pretrained True \
--network.checkpoint 'results/cifar10_csinet_csi_e100_lr0.1/best.ckpt' \
--save_output True
