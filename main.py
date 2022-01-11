from openood.pipelines import get_pipeline
from openood.utils import setup_config

if __name__ == '__main__':
    """Main entrance. Config is all you need to provide to run the code. Config
    should be provided in the format of YAML and can be modified with command
    line.

    Example:
        python main.py \
            --config configs/datasets/mnist_datasets.yml \
            configs/train/mnist_baseline.yml \
            --dataset.image_size 32 \
            --network res18

    Note:
        A config file is the minimum requirement.
        You don't need to add "--config_key new_value"
        if you don't have anything to modify.
    """

    # read command line to composite configs.
    config = setup_config()

    # run!
    pipeline = get_pipeline(config)
    pipeline.run()
