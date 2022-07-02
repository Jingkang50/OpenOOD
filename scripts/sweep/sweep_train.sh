# sh ./scripts/sweep/sweep_train.sh
python ./scripts/sweep/sweep_train.py \
--benchmarks 'tin20' \
--launcher 'slurm'
