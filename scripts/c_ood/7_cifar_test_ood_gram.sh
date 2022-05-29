#!/bin/bash
# sh scripts/c_ood/7_cifar_test_ood_gram.sh

GPU=1
CPU=1
node=36
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
'''
python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/datasets/objects/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_gram.yml \
configs/postprocessors/gram.yml \
--dataset.image_size 32 \
--network.name resnet18_32x32 \
--num_workers 8
'''
'''
python main.py \
--config configs/datasets/digits/mnist.yml \
configs/datasets/digits/mnist_ood.yml \
configs/networks/lenet.yml \
configs/pipelines/test/test_gram.yml \
configs/postprocessors/gram.yml \
configs/preprocessors/base_preprocessor.yml
'''
python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/datasets/objects/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_gram.yml \
configs/postprocessors/gram.yml \
configs/preprocessors/base_preprocessor.yml
