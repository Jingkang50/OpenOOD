#!/bin/bash
# ==============================================================================
# KPCA OOD Detection Evaluation on ImageNet-1K
#
# Hyperparameters are mapped from the original KPCA papers:
#   Fang et al., "Kernel PCA for OOD Detection", NEURIPS 2024.
#   Official repo: https://github.com/fanghenshaometeor/ood-kernel-pca
#   Fang et al., "Kernel PCA for OOD Detection: Non-Linear Kernel Selection
#   and Approximation", TPAMI 2026.
#   Official repo: https://github.com/fanghenshaometeor/ood-kernel-pca-ext
#
# Usage:
#   bash scripts/ood/kpca/imagenet_test_ood_kpca.sh
# ==============================================================================

# ==============================================================================
# We strongly recommend referring to the official KPCA repo 
#   at https://github.com/fanghenshaometeor/ood-kernel-pca-ext
# This official version is more readable, flexible, computationally efficient, and user-friendly for reproduction.
#   It also provides more options, including support for MobileNet and ReAct fusion.
# ==============================================================================

# -------------------------------------------------------------------
# ResNet-50, standard (cross-entropy) training
# -------------------------------------------------------------------

# Nystrom approximation
#   Original: --approx NYS --gamma 0.15 --M 2048 --exp_var_ratio 0.995
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/kpca.yml \
    --postprocessor.name kpca \
    --postprocessor.postprocessor_args.approx NYS \
    --postprocessor.postprocessor_args.gamma 0.15 \
    --postprocessor.postprocessor_args.M 2048 \
    --postprocessor.postprocessor_args.exp_var_ratio 0.995 \
    --network.pretrained True \
    --network.checkpoint ./checkpoints/resnet50-0676ba61.pth \
    --merge_option merge \
    --num_workers 8 \
    --dataset.train.batch_size 256

# Random Fourier Features (RFF) approximation
#   Original: --approx RFF --gamma 3 --M 4096 --exp_var_ratio 0.5
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/kpca.yml \
    --postprocessor.name kpca \
    --postprocessor.postprocessor_args.approx RFF \
    --postprocessor.postprocessor_args.gamma 3 \
    --postprocessor.postprocessor_args.M 4096 \
    --postprocessor.postprocessor_args.exp_var_ratio 0.5 \
    --network.pretrained True \
    --network.checkpoint ./checkpoints/resnet50-0676ba61.pth \
    --merge_option merge \
    --num_workers 8 \
    --dataset.train.batch_size 256

