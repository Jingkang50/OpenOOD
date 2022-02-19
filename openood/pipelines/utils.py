from openood.utils import Config

from .ad_test_det import ADTestPipeline
from .feat_extract_pipeline import FeatExtractPipeline
from .test_acc_pipeline import TestAccPipeline
from .test_ood_pipeline import TestOODPipeline
from .train_ad_pipeline import TrainADPipeline
from .train_pipeline import TrainPipeline


def get_pipeline(config: Config):
    pipelines = {
        'train': TrainPipeline,
        'test_acc': TestAccPipeline,
        'feat_extract': FeatExtractPipeline,
        'test_ood': TestOODPipeline,
        'kdad_test_det': ADTestPipeline,
        'kdad_train': TrainADPipeline
    }

    return pipelines[config.pipeline.name](config)
