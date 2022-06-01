from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.preprocessors.utils import get_preprocessor
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger


class TrainAdPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)
        train_loader = id_loader_dict['train']

        # init network
        net = get_network(self.config.network)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, self.config)
        evaluator = get_evaluator(self.config)

        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict, ood_loader_dict)

        # init recorder
        recorder = get_recorder(self.config)

        print('Start training...', flush=True)
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            # train the model
            net, train_metrics = trainer.train_epoch(epoch_idx)
            test_metrics = evaluator.eval_ood(net,
                                              id_loader_dict,
                                              ood_loader_dict,
                                              postprocessor=postprocessor,
                                              epoch_idx=epoch_idx)
            # save model and report the result
            recorder.save_model(net, test_metrics)
            recorder.report(train_metrics, test_metrics)
        recorder.summary()

        # evaluate on test set
        print('Start testing...', flush=True)
        test_metrics = evaluator.eval_ood(net,
                                          id_loader_dict,
                                          ood_loader_dict,
                                          postprocessor=postprocessor)
        evaluator.report(test_metrics)
