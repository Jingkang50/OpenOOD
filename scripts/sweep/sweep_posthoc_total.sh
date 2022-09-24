# sh ./scripts/sweep/sweep_posthoc_total.sh
python ./scripts/sweep/sweep_posthoc.py \
--benchmarks 'cifar6' 'cifar50' 'mnist6' 'tin20' 'cifar10' 'cifar100' \
--methods 'msp' 'odin' 'mds' 'gram' 'ebo' 'gradnorm' 'react' 'dice' 'vim' 'mls' 'klm' 'knn' \
--metrics 'osr' 'ood'  \
--metric2save 'fpr95' 'auroc' 'aupr_in' \
--output-dir './results/total' \
--launcher 'local' \
--merge-option 'pass' \
--update_form_only
