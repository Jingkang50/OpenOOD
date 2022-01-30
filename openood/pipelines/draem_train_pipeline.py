from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.preprocessors.utils import get_preprocessor
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger


class DRAEMTrainPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get preprocessor
        preprocessor = get_preprocessor(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config.dataset, preprocessor)
        train_loader = loader_dict['train']
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, self.config)
        evaluator = get_evaluator(self.config)

        # init recorder
        recorder = get_recorder(self.config)

        print('Start training...', flush=True)
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            # train the model
            net, train_metrics = trainer.train_epoch(epoch_idx)
            test_metrics = evaluator.eval(net,
                                          test_loader,
                                          epoch_idx=epoch_idx)
            # save model and report the result
            recorder.save_model(net, test_metrics)
            recorder.report(train_metrics, test_metrics)
        recorder.summary()

        # evaluate on test set
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
