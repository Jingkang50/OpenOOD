from openood.pipelines.finetune_pipeline import FinetunePipeline
from openood.utils import Config

from .feat_extract_pipeline import FeatExtractPipeline
from .test_acc_pipeline import TestAccPipeline
from .test_ad_pipeline import TestAdPipeline
from .test_ood_pipeline import TestOODPipeline
from .train_ad_pipeline import TrainAdPipeline
from .train_dsvdd_pipeline import TrainDSVDDPipeline
from .train_pipeline import TrainPipeline


def get_pipeline(config: Config):
    pipelines = {
        'train': TrainPipeline,
        'finetune': FinetunePipeline,
        'test_acc': TestAccPipeline,
        'feat_extract': FeatExtractPipeline,
        'test_ood': TestOODPipeline,
        'test_ad': TestAdPipeline,
        'train_ad': TrainAdPipeline,
        'train_dsvdd': TrainDSVDDPipeline
    }

    return pipelines[config.pipeline.name](config)
