from openood.datasets import get_dataloader
from openood.evaluation import get_evaluator
from openood.networks import get_network
from openood.utils import setup_logger


class TestAccPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config.dataset)
        val_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)

        # init evaluator
        evaluator = get_evaluator(self.config)

        # start calculating accuracy
        print('Start evaluation...', flush=True)
        val_metrics = evaluator.eval_acc(net, val_loader, -1)
        print('Complete Evaluation, accuracy {:.2f}%'.format(
            100 * val_metrics['test_accuracy']),
              flush=True)
