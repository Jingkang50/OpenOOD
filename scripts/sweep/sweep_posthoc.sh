# sh ./scripts/sweep/sweep_posthoc.sh
python ./scripts/sweep/sweep_posthoc.py \
--benchmarks 'tin20' \
--methods 'msp' \
--metrics 'osr' \
--output-dir './results/ood' \
--launcher 'local'
