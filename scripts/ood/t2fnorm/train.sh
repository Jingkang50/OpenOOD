#!/bin/bash
# sh scripts/basics/cifar10/train_cifar10.sh
idx=-1
for j in {1..5}
do
    datasets=("cifar10" "cifar100")
    for dataset in "${datasets[@]}"
    do
        networks=("resnet18_32x32")
        for network in "${networks[@]}"
        do
            idx=$(( (idx + 1) % 4 ))
            CUDA_VISIBLE_DEVICES=$idx python main.py \
                                      --config configs/datasets/$dataset/$dataset.yml \
                                      configs/datasets/$dataset/${dataset}_ood.yml \
                                      configs/networks/$network.yml \
                                      configs/preprocessors/base_preprocessor.yml \
                                      configs/pipelines/train/train_t2fnorm.yml \
                                      --network.pretrained False \
                                      --dataset.image_size 32 \
                                      --optimizer.num_epochs 100 \
                                      --num_workers 8 \
                                      --seed $RANDOM \
                                      --network.tau 0.1 &
        done
    done
done
