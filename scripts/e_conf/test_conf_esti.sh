#!/bin/bash
# sh scripts/e_conf/test_conf_esti.sh

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/digits/mnist.yml \
configs/datasets/digits/mnist_ood.yml \
configs/pipelines/test/test_conf_esti.yml \
configs/networks/conf_net.yml \
configs/preprocessors/conf_esti_preprocessor.yml
