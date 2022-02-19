from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger


class TrainADPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # attain id and ood dataset
        id_loader_dict = get_dataloader(self.config.dataset)
        ood_loader_dict = get_ood_dataloader(self.config.ood_dataset)

        # attain source net and clone net
        net = get_network(self.config.network)

        # attain trainer
        trainer = get_trainer(net, id_loader_dict, self.config)

        # attain postprocessor but useless
        postprocessor = get_postprocessor(self.config)

        # attain evaluator
        evaluator = get_evaluator(self.config)

        # attain recorder
        recorder = get_recorder(self.config)

        print('Start training...', flush=True)
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            net, train_metrics = trainer.train_epoch(epoch_idx)
            eval_metrics = evaluator.eval_ood(net, id_loader_dict,
                                              ood_loader_dict, postprocessor,
                                              epoch_idx)
            recorder.save_model(net, eval_metrics)
            recorder.report(train_metrics, eval_metrics)
        recorder.summary()

        # start ood detection test
        print('Start testing...', flush=True)
        test_metrics = evaluator.eval_ood(net, id_loader_dict, ood_loader_dict,
                                          postprocessor)
        print('\nComplete Evaluation, {}: {:.2f}'.format(
            self.config.metrics, 100.0 * test_metrics[self.config.metrics]),
              flush=True)
        print('Completed!', flush=True)
