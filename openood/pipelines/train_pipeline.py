from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger


class TrainPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config.dataset)
        train_loader, val_loader = loader_dict['train'], loader_dict['test']

        # init network
        net = get_network(self.config.network)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, self.config)
        evaluator = get_evaluator(self.config)

        # init recorder
        recorder = get_recorder(self.config)

        print('Start training...', flush=True)
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            # train and eval the model
            net, train_metrics = trainer.train_epoch(epoch_idx)
            val_metrics = evaluator.eval_classification(
                net, val_loader, epoch_idx)
            # save model and report the result
            recorder.save_model(net, val_metrics)
            recorder.report(train_metrics, val_metrics)

        recorder.summary()
