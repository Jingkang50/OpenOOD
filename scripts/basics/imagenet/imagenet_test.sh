#!/bin/bash
# sh scripts/c_ood/-1_imagenet_test.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/imagenet/imagenet.yml \
configs/networks/resnet50.yml \
configs/pipelines/test/test_acc.yml \
--num_workers 20 \
--dataset.test.batch_size 512 \
--dataset.val.batch_size 512 \
--network.pretrained True \
--network.checkpoint "./checkpoints/resnet50-0676ba61-pytorch.pth" \
--save_output True \
--num_gpus 1
