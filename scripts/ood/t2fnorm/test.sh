#!/bin/bash
# sh scripts/basics/cifar10/test_cifar10.sh
idx=-1
datasets=("cifar10" "cifar100")
pids=()
counter=0
for dataset in "${datasets[@]}"
do
    networks=("resnet18_32x32")
    for network in "${networks[@]}"
    do
        folders=( $(find ./results/t2fnorm/models/ -maxdepth 1 -type d -name "*${dataset}_${network}*") )
        if [ ${#folders[@]} -ne 5 ]; then
            echo "Should contain 5 folders"
            echo ${#folders[@]}
            #exit 1
        fi
        for folder in "${folders[@]}"
        do
            postprocessors=("base" "ebo" "dice" "gradnorm" "odin")
            for postproc in "${postprocessors[@]}"
            do
                idx=$(( (idx + 1) % 8 ))
                CUDA_VISIBLE_DEVICES=$idx python main.py \
                --config configs/datasets/$dataset/$dataset.yml \
                configs/datasets/$dataset/${dataset}_ood.yml \
                configs/networks/$network.yml \
                configs/pipelines/test/test_ood.yml \
                configs/preprocessors/base_preprocessor.yml \
                configs/postprocessors/$postproc.yml \
                --num_workers 8 \
                --network.checkpoint "${folder}/best.ckpt" \
                --mark T2FNorm &
                pids+=($!)
                counter=$((counter+1))
                if [ $counter -eq 8 ]; then
                    for pid in "${pids[@]}"; do
                        wait $pid
                    done
                    pids=()
                    counter=0
                fi
            done
        done
    done
done

for pid in "${pids[@]}"; do
    wait $pid
done
