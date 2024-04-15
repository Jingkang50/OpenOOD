python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/networks/convnext_v2.yml \
    configs/pipelines/train/train_sweep.yml \
    configs/preprocessors/base_preprocessor.yml \
    --num_workers 1 \
    --mark 0
