GPU=1
CPU=1
node=30
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/OpenMax/cifar100.yml \
configs/networks/openmax.yml \
configs/pipelines/train/train_openmax.yml \
configs/postprocessors/openmax.yml \
configs/preprocessors/None_preprocessor.yml
--preprocessor.name None
--num_workers 4