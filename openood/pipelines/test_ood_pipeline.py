from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger


class TestOODPipeline:
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

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)

        # start calculating accuracy
        print('Start evaluation...', flush=True)
        acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'])
        print('Accuracy {:.2f}%'.format(100 * acc_metrics['acc']), flush=True)
        print(u'\u2500' * 70, flush=True)

        # start evaluating ood detection methods
        evaluator.eval_ood(net, id_loader_dict, ood_loader_dict, postprocessor)
        print('Completed!', flush=True)
