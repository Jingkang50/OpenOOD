# sh scripts/osr/openmax/cifar10_train_openmax.sh
GPU=1
CPU=1
node=30
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/networks/openmax.yml \
configs/pipelines/train/train_openmax.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/openmax.yml \
--num_workers 8
