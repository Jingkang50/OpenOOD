echo 'Running kl_matching'
#!/bin/bash
# sh scripts/ood/kl_matching/cifar10_test_ood_kl_matching.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/klm.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0


echo 'Running rankfeat'
#!/bin/bash
# sh scripts/ood/rankfeat/cifar10_test_ood_rankfeat.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/rankfeat.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 1


echo 'Running ash'
#!/bin/bash
# sh scripts/ood/ash/cifar10_test_ood_ash.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ash.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 1


echo 'Running gen'
#!/bin/bash
# sh scripts/ood/she/cifar10_test_ood_she.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/gen.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 1


echo 'Running scale'
#!/bin/bash
# sh scripts/ood/scale/cifar10_test_ood_scale.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/scale.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 1


echo 'Running residual'
#!/bin/bash
# sh scripts/ood/residual/cifar10_test_ood_residual.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
--config configs/datasets/tiny_inet/tiny_inat.yml \
configs/datasets/tiny_inet/tiny_inat_ood.yml \
configs/networks/convnext_base.yml \
configs/pipelines/test/test_hc_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/residual.yml \
--num_workers 1 \
--network.checkpoint 'pretrained_models/checkpoint-best.pth' \
--mark 0

echo 'Running gradnorm'
#!/bin/bash
# sh scripts/ood/gradnorm/cifar10_test_ood_gradnorm.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/gradnorm.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0


echo 'Running ebo'
#!/bin/bash
# sh scripts/ood/ebo/cifar10_test_ood_ebo.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 1 \
    --postprocessor.postprocessor_args.temperature 1


echo 'Running mls'
#!/bin/bash
# sh scripts/ood/mls/cifar10_test_ood_maxlogit.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/mls.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0


echo 'Running vim'
#!/bin/bash
# sh scripts/ood/vim/cifar10_test_ood_vim.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/vim.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0 \
    --postprocessor.postprocessor_args.dim 256


echo 'Running oe'
#!/bin/bash
# sh scripts/ood/oe/cifar10_test_oe.sh

GPU=1
CPU=1
node=63
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0


echo 'Running mds'
#!/bin/bash
# sh scripts/ood/mds/cifar10_test_ood_mds.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/mds.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 1 \
    --postprocessor.postprocessor_args.temperature 1


echo 'Running vos'
#!/bin/bash
# sh scripts/ood/vos/cifar10_test_vos.sh

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 1 \
    --network.pretrained True \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark vos


echo 'Running dice'
#!/bin/bash
# sh scripts/ood/dice/cifar10_test_ood_dice.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/dice.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0


echo 'Running relation'
#!/bin/bash
# sh scripts/ood/relation/cifar10_test_ood_relation.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/relation.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0


echo 'Running gram'
#!/bin/bash
# sh scripts/ood/gram/cifar10_test_ood_gram.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/gram.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0


echo 'Running msp'
#!/bin/bash
# sh scripts/ood/msp/cifar10_test_ood_msp.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0 \
    --merge_option merge


echo 'Running knn'
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

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/knn.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0


echo 'Running odin'
#!/bin/bash
# sh scripts/ood/odin/cifar10_test_ood_odin.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/odin.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0


echo 'Running she'
#!/bin/bash
# sh scripts/ood/she/cifar10_test_ood_she.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/she.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 1


echo 'Running rmds'
#!/bin/bash
# sh scripts/ood/rmds/cifar10_test_ood_rmds.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/rmds.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 1


echo 'Running sem'
#!/bin/bash
# sh scripts/ood/sem/cifar10_test_ood_sem.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/tiny_inet/tiny_inat.yml \
configs/datasets/tiny_inet/tiny_inat_ood.yml \
configs/networks/convnext_base.yml \
configs/pipelines/test/test_hc_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/gmm.yml \
--num_workers 1 \
--network.checkpoint 'pretrained_models/checkpoint-best.pth' \
--mark no_train

echo 'Running mds_ensemble'
#!/bin/bash
# sh scripts/ood/mds_ensemble/cifar10_test_ood_mds_ensemble.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_hc_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/mds_ensemble.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0


