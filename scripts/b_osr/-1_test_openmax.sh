# sh scripts/b_osr/-1_test_openmax.sh
GPU=1
CPU=1
node=30
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/OpenMax/cifar100.yml \
configs/networks/test_openmax.yml \
configs/pipelines/test/test_openmax.yml \
configs/postprocessors/openmax.yml \
configs/preprocessors/None_preprocessor.yml \
--preprocessor.name None \
--num_workers 4