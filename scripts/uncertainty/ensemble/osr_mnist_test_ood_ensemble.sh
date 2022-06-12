#!/bin/bash
# sh scripts/uncertainty/ensemble/osr_mnist_test_ood_ensemble.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/osr_mnist6/mnist6_seed1.yml \
configs/datasets/osr_mnist6/mnist6_seed1_ood.yml \
configs/networks/lenet.yml \
configs/pipelines/test/test_osr.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/ensemble.yml \
--network.pretrained False \
--num_workers 8 \
--mark 0 \
--postprocessor.postprocessor_args.network_name lenet \
--postprocessor.postprocessor_args.checkpoint_root 'results/_osr_mnist6_test_ensemble' \
--postprocessor.postprocessor_args.num_networks 5 \
--dataset.test.batch_size 64 \
--dataset.val.batch_size 64 \
--ood_dataset.batch_size 64
