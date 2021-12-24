from openood.datasets import get_dataloader
from openood.evaluation import get_evaluator
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
        train_loader, val_loader = get_dataloader(self.config)

        # init network
        net = get_network(self.config)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, self.config)
        evaluator = get_evaluator(net, val_loader, self.config)

        # init recorder
        recorder = get_recorder(self.config)

        for epoch in range(self.config.optimizer.epochs):
            # train and eval the model
            net, train_metrics = trainer.train_epoch()
            net, val_metrics = evaluator.eval()

            # save model and report the result
            recorder.save_best_model(net, val_metrics, epoch)
            recorder.report(net, train_metrics, val_metrics, epoch)
