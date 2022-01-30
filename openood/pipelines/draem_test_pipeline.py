from openood.datasets import get_dataloader
from openood.evaluators.utils import get_evaluator
from openood.networks.utils import get_network
from openood.preprocessors import get_preprocessor
from openood.utils import setup_logger


class DRAEMTestPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get preprocessor
        preprocessor = get_preprocessor(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config.dataset, preprocessor)
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)

        # init DRAEM evaluator
        evaluator = get_evaluator(self.config)

        print('Start testing...', flush=True)
        test_metrics = evaluator.eval(net, test_loader)
        print('Complete Evaluation:\n'
              '{}\n'
              '==============================\n'
              'AUC Image: {:.2f} \nAP Image: {:.2f} \n'
              'AUC Pixel: {:.2f} \nAP Pixel: {:.2f} \n'
              '=============================='.format(
                  self.config.dataset.name, 100.0 * test_metrics['image_auc'],
                  100.0 * test_metrics['image_ap'],
                  100.0 * test_metrics['pixel_auc'],
                  100.0 * test_metrics['pixel_ap']),
              flush=True)
        print('Completed!', flush=True)
