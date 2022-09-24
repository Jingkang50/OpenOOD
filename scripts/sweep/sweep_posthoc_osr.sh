# sh ./scripts/sweep/sweep_posthoc_osr.sh
python ./scripts/sweep/sweep_posthoc.py \
--benchmarks 'cifar6' 'cifar50' 'mnist6' 'tin20' \
--methods 'msp' \
--metrics 'osr' \
--metric2save 'fpr95' 'auroc' 'aupr_in' \
--output-dir './results/osr' \
--launcher 'local' \
--update_form_only
