from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.utils import get_evaluator
from openood.networks.utils import get_network
from openood.preprocessors import get_preprocessor
from openood.utils import setup_logger


class TestAdPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get preprocessor
        preprocessor = get_preprocessor(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config.dataset, preprocessor)
        ood_loader_dict = get_ood_dataloader(self.config.ood_dataset,
                                             preprocessor)

        # init network
        net = get_network(self.config.network)

        # init DRAEM evaluator
        evaluator = get_evaluator(self.config)

        print('Start testing...', flush=True)
        test_metrics = evaluator.eval_ood(net, id_loader_dict, ood_loader_dict)
        evaluator.report(test_metrics)
