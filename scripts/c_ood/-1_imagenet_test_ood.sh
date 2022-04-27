#!/bin/bash
# sh scripts/c_ood/-1_imagenet_test_ood.sh

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
configs/datasets/digits/mnist_ood.yml \
configs/networks/reactnet.yml \
configs/pipelines/test/test_ood.yml \
configs/postprocessors/react.yml \
--num_workers 20 \
--ood_dataset.image_size 256 \
--dataset.val.batch_size 256 \
--network.pretrained False \
--save_output True
