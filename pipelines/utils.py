from openood.utils import Config

from .feat_extract_pipeline import FeatExtractPipeline
from .finetune_pipeline import FinetunePipeline
from .test_acc_pipeline import TestAccPipeline
from .test_ad_pipeline import TestAdPipeline
from .test_ood_pipeline import TestOODPipeline
from .test_patchcore_pipeline import TestPatchcorePipeline
from .train_ad_pipeline import TrainAdPipeline
from .train_arplgan_pipeline import TrainARPLGANPipeline
from .train_dsvdd_pipeline import TrainDSVDDPipeline
from .train_opengan_pipeline import TrainOpenGanPipeline
from .train_pipeline import TrainPipeline


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
        'train_dsvdd': TrainDSVDDPipeline,
        'train_opengan': TrainOpenGanPipeline,
        'train_arplgan': TrainARPLGANPipeline,
    }

    return pipelines[config.pipeline.name](config)
