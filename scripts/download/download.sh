# sh ./scripts/download/download.sh
python ./scripts/download/download.py \
--contents 'datasets' \
--datasets 'imagenet_1k' 'misc' \
--checkpoints 'all' \
--save_dir './data' './results' \
--dataset_mode 'benchmark'
