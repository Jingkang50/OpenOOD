from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger


class ADTestPipeline:
    def __init__(self, config):
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # attain test dataset
        id_loader_dict = get_dataloader(self.config.dataset)
        ood_loader_dict = get_ood_dataloader(self.config.ood_dataset)
        net = get_network(self.config.network)

        # attain evaluator
        evaluator = get_evaluator(self.config)

        # attain postprocessor but useless
        postprocessor = get_postprocessor(self.config)

        # start evaluating ood detection methods
        print('Start testing...', flush=True)
        test_metrics = evaluator.eval_ood(net, id_loader_dict, ood_loader_dict,
                                          postprocessor)
        print('\nComplete Evaluation, {} {:.2f}'.format(
            self.config.metrics, 100.0 * test_metrics[self.config.metrics]),
              flush=True)
        print('Completed!', flush=True)
