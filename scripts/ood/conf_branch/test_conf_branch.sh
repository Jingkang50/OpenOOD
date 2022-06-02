#!/bin/bash
# sh scripts/e_conf/test_conf_esti.sh

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
'''
python main.py \
--config configs/datasets/Conf_Esti/svhn.yml \
configs/datasets/Conf_Esti/conf_esti_ood.yml \
configs/pipelines/test/test_conf_esti.yml \
configs/networks/conf_esti.yml \
configs/preprocessors/conf_esti_preprocessor.yml
'''
#Mnist Lenet
python main.py \
--config configs/datasets/digits/mnist.yml \
configs/datasets/digits/mnist_ood.yml \
configs/pipelines/test/test_ood.yml \
configs/networks/conf_net.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/conf.yml
