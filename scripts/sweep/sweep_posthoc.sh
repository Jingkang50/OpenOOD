# sh ./scripts/sweep/sweep_posthoc.sh
python ./scripts/sweep/sweep_posthoc.py \
--benchmarks 'mnist' \
--methods 'msp' \
--metrics 'ood' \
--output-dir './results/ood' \
--launcher 'local' \
