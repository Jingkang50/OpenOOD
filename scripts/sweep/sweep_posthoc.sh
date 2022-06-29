# sh ./scripts/sweep/sweep_posthoc.sh
python ./scripts/sweep/sweep_posthoc.py \
--benchmarks 'cifar6' \
--methods 'msp' \
--metrics 'ood' \
--metric2save 'auroc' \
--output-dir './results/ood' \
--launcher 'slurm' \
