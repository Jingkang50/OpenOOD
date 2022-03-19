from openood.pipelines.finetune_pipeline import FinetunePipeline
from openood.utils import Config

from .feat_extract_pipeline import FeatExtractPipeline
from .test_acc_pipeline import TestAccPipeline
from .test_ad_pipeline import TestAdPipeline
from .test_ood_pipeline import TestOODPipeline
from .train_ad_pipeline import TrainAdPipeline
from .train_pipeline import TrainPipeline
from .test_patchcore_pipeline import TestPatchcorePipeline


def get_pipeline(config: Config):
    pipelines = {
        'train': TrainPipeline,
        'finetune': FinetunePipeline,
        'test_acc': TestAccPipeline,
        'feat_extract': FeatExtractPipeline,
        'test_ood': TestOODPipeline,
        'test_patchcore': TestPatchcorePipeline,
        'test_ad': TestAdPipeline,
        'train_ad': TrainAdPipeline,
    }

    return pipelines[config.pipeline.name](config)
