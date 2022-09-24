# sh ./scripts/download/dowanload.sh
python ./scripts/download/dowanload.py \
--contents 'datasets' 'checkpoints' \
--datasets 'default' \
--checkpoints 'all' \
--save_dir './data' './results' \
--dataset_mode 'benchmark'
