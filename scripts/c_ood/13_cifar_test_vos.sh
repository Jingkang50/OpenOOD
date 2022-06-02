#!/bin/bash
# sh scripts/c_ood/13_cifar_test_vos.sh

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/VOS/cifar10.yml \
configs/datasets/VOS/vos_ood.yml \
configs/pipelines/test/test_vos.yml \
configs/networks/vos_net.yml \
configs/preprocessors/base_preprocessor.yml
