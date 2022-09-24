# sh ./scripts/sweep/sweep_posthoc-backup.sh
python ./scripts/sweep/sweep_posthoc.py \
--benchmarks 'cifar10' 'cifar100' \
--methods 'msp' 'odin' 'mds' 'gram' 'ebo' 'gradnorm' 'react' 'dice' 'vim' 'mls' 'klm' 'knn' \
--metrics 'ood' \
--metric2save 'fpr95' 'auroc' 'aupr_in' \
--output-dir './results/ood' \
--launcher 'local' \
--update_form_only
