#!/bin/bash
# sh scripts/a_anomaly/4_draem_train_cifar.sh

# GPU=1
# CPU=1
# node=30
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/datasets/objects/cifar10_ood.yml \
configs/networks/draem.yml \
configs/pipelines/train/train_draem.yml \
configs/preprocessors/draem_preprocessor.yml \
configs/postprocessors/msp.yml \
--evaluator.name ad \
--dataset.train.batch_size 64 \
--postprocessor.name dream \
--optimizer.num_epochs 100
