# sh ./scripts/sweep/sweep_posthoc.sh
python ./scripts/sweep/sweep_posthoc.py \
--benchmarks 'cifar10' \
--methods 'msp' \
--metrics 'ood' \
--metric2save 'fpr95' 'auroc' 'aupr_in' \
--output-dir './results/ood' \
--launcher 'local' \
--update_form_only
