python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/conf_branch.yml \
    configs/pipelines/train/train_conf_branch.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.backbone.name resnet18_224x224 \
    --optimizer.num_epochs 90 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0
