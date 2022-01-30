from openood.pipelines.finetune_pipeline import FinetunePipeline
from openood.utils import Config

from .draem_test_pipeline import DRAEMTestPipeline
from .draem_train_pipeline import DRAEMTrainPipeline
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
        'test_DRAEM': DRAEMTestPipeline,
        'train_DRAEM': DRAEMTrainPipeline,
    }

    return pipelines[config.pipeline.name](config)
