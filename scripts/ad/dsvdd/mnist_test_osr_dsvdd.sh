#!/bin/bash
# sh scripts/ad/dsvdd/mnist_test_osr_dsvdd.sh


PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/osr_mnist6/mnist6_seed1.yml \
configs/datasets/osr_mnist6/mnist6_seed1_ood.yml \
configs/networks/lenet.yml \
configs/pipelines/test/test_dsvdd.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/dsvdd.yml \
--network.pretrained True \
--network.checkpoint 'results/osr_mnist6_seed1_lenet_dsvdd_e100/best.ckpt' \
--evaluator.name osr
