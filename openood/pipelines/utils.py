from openood.pipelines.finetune_pipeline import FinetunePipeline
from openood.utils import Config

from .ad_test_pipeline import AdTestPipeline
from .ad_train_pipeline import AdTrainPipeline
from .feat_extract_pipeline import FeatExtractPipeline
from .test_acc_pipeline import TestAccPipeline
from .test_ood_pipeline import TestOODPipeline
from .train_pipeline import TrainPipeline


def get_pipeline(config: Config):
    pipelines = {
        'train': TrainPipeline,
        'finetune': FinetunePipeline,
        'test_acc': TestAccPipeline,
        'feat_extract': FeatExtractPipeline,
        'test_ood': TestOODPipeline,
        'test_ad': AdTestPipeline,
        'train_ad': AdTrainPipeline,
    }

    return pipelines[config.pipeline.name](config)
