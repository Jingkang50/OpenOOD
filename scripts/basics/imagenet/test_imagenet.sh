#!/bin/bash
# sh scripts/basics/imagenet/test_imagenet.sh

GPU=1
CPU=1
node=76
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/imagenet/imagenet.yml \
configs/networks/resnet50.yml \
configs/pipelines/test/test_acc.yml \
configs/preprocessors/base_preprocessor.yml \
--num_workers 20 \
--dataset.test.batch_size 512 \
--dataset.val.batch_size 512 \
--network.pretrained True \
--network.checkpoint "./results/checkpoints/imagenet_res50_acc76.10.pth" \
--save_output True \
--num_gpus 1
