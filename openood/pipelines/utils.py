from typing import Any, Dict

from openood.utils import Config

from .train_pipeline import TrainPipeline


def get_pipeline(config: Config, ):
    pipelines = {
        'train': TrainPipeline,
    }

    return pipelines[config.pipeline.name](config)
