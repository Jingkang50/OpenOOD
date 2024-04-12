#!/bin/bash
# sh scripts/ood/knn/cifar10_test_ood_knn.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
# load config files for cifar10 baseline

#!/bin/bash

# Define an array of postprocessors
# postprocessors=(ash conf_branch dice gen gradnorm ish mcd mds_ensemble mls odin rankfeat relation rmds scale she t2fnorm vim cider csi ebo godin gram kl_matching logitnorm mds mixoe mos npos oe react residual rotpred sem ssd udg vos)
postprocessors=( odin )

# Loop through each postprocessor and execute the command
for postprocessor in "${postprocessors[@]}"
do
    echo "Running with postprocessor: $postprocessor"
    python main.py \
        --config configs/datasets/tiny_inet/tiny_inat.yml \
        configs/datasets/tiny_inet/tiny_inat_ood.yml \
        configs/networks/convnext_base.yml \
        configs/pipelines/test/test_ood.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/$postprocessor.yml \
        --num_workers 1 \
        --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
        --mark 0
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "$postprocessor: done"
    else
        echo "$postprocessor: failed"
    fi

done

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
# python scripts/eval_ood.py \
#    --id-data cifar10 \
#    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
#    --postprocessor knn \
#    --save-score --save-csv
