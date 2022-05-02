from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger


class TestPatchcorePipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config.dataset)

        # init network
        net = get_network(self.config.network)
        
        # init ood evaluator
        evaluator = get_evaluator(self.config)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)
        
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict)
        print(u'\u2500' * 70, flush=True)

        # start calculating accuracy
        print('Start evaluation...', flush=True)
        acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'],
                                         postprocessor)
        
        print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
              flush=True)
        print(u'\u2500' * 70, flush=True)

        print('Completed!', flush=True)
        
