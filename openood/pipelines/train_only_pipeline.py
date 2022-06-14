from openood.datasets import get_feature_dataloader
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger


class TrainOpenGanPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        feat_loader = get_feature_dataloader(self.config.dataset)

        # init network
        net = get_network(self.config.network)

        # init trainer
        trainer = get_trainer(net, feat_loader, self.config)

        # init recorder
        recorder = get_recorder(self.config)

        print('Start training...', flush=True)
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            # train the model
            net, train_metrics = trainer.train_epoch(epoch_idx)
            recorder.save_model(net, train_metrics)
            recorder.report(train_metrics)
        recorder.summary()

        print('Completed!', flush=True)
