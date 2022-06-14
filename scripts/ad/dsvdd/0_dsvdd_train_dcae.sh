#!/bin/bash
# sh scripts/a_anomaly/0_dsvdd_train_dcae.sh


PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/datasets/objects/cifar10_ood.yml \
configs/pipelines/train/train_dsvdd.yml \
configs/networks/dsvdd.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
--pipeline.name train_ad \
--postprocessor.name dsvdd \
--evaluator.name ad \
--recorder.name ad \
--optimizer.num_epochs 2 \
--network.pretrained True \
--network.checkpoint 'results/cifar10_dcae_dsvdd_e3/best.ckpt'
