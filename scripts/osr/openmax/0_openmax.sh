# sh scripts/b_osr/0_openmax.sh
GPU=1
CPU=1
node=30
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/digits/mnist.yml \
configs/networks/openmax.yml \
configs/pipelines/train/train_openmax.yml \
configs/postprocessors/openmax.yml \
configs/preprocessors/base_preprocessor.yml \
--num_workers 4
