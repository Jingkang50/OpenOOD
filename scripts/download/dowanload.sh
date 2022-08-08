# sh ./scripts/download/dowanload.sh
python ./scripts/download/dowanload.py \
--contents 'datasets' 'checkpoints' \
--datasets 'mnist' 'cifar-10' \
--checkpoints 'mnist_lenet' 'cifar10_res18' \
--save_dir './data' './results' \
--dataset_mode 'benchmark'
