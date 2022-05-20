from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.utils import setup_logger


class TestMOSPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config.dataset)
        ood_loader_dict = get_ood_dataloader(self.config.ood_dataset)

        # init network
        net = get_network(self.config.network)

        # init ood evaluator
        evaluator = get_evaluator(self.config)
        evaluator.eval_ood(net, id_loader_dict, ood_loader_dict)
        print('Completed!', flush=True)
