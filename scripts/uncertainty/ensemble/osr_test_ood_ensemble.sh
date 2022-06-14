#!/bin/bash
# sh scripts/uncertainty/ensemble/osr_test_ood_ensemble.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/osr_tin20/tin20_seed1.yml \
configs/datasets/osr_tin20/tin20_seed1_ood.yml \
configs/networks/resnet18_64x64.yml \
configs/pipelines/test/test_osr.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/ensemble.yml \
--network.pretrained False \
--num_workers 8 \
--mark 0 \
--postprocessor.postprocessor_args.network_name resnet18_64x64 \
--postprocessor.postprocessor_args.checkpoint_root 'results/osr_tin20_seed1' \
--postprocessor.postprocessor_args.num_networks 5 \
--dataset.test.batch_size 64 \
--dataset.val.batch_size 64 \
--ood_dataset.batch_size 64
