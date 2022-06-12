!/bin/bash
# sh scripts/uncertainty/pixmix/mnist_test_ood_pixmix.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/lenet.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
--num_workers 8 \
--network.checkpoint 'results/mnist_lenet_base_e100_lr0.1_pixmix/best.ckpt' \
--mark pixmix
