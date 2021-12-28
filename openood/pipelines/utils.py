from openood.utils import Config

from .feat_extract_pipeline import FeatExtractPipeline
from .test_acc_pipeline import TestAccPipeline
from .train_pipeline import TrainPipeline


def get_pipeline(config: Config):
    pipelines = {
        'train': TrainPipeline,
        'test_acc': TestAccPipeline,
        'feat_extract': FeatExtractPipeline
    }

    return pipelines[config.pipeline.name](config)
