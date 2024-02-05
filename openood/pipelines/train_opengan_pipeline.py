import numpy as np
import torch

from openood.datasets import get_feature_opengan_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger


class TrainOpenGanPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # get dataloader
        dataloaders = get_feature_opengan_dataloader(self.config.dataset)
        id_loaders = {
            'train': dataloaders['id_train'],
            'val': dataloaders['id_val']
        }  # just for consistency with evaluator
        ood_loaders = {'val': dataloaders['ood_val']}

        # init network
        net = get_network(self.config.network)
        if self.config.num_gpus * self.config.num_machines > 1:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

        # init trainer
        trainer = get_trainer(net, dataloaders['id_train'],
                              dataloaders['id_val'], self.config)
        evaluator = get_evaluator(self.config)

        # init recorder
        recorder = get_recorder(self.config)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)

        print('Start training...', flush=True)
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            if isinstance(dataloaders['id_train'].sampler,
                          torch.utils.data.distributed.DistributedSampler):
                dataloaders['id_train'].sampler.set_epoch(epoch_idx - 1)

            # train the model
            net, train_metrics = trainer.train_epoch(epoch_idx)
            val_metrics = evaluator.eval_ood_val(net, id_loaders, ood_loaders,
                                                 postprocessor)
            val_metrics['epoch_idx'] = train_metrics['epoch_idx']
            recorder.save_model(net, val_metrics)
            recorder.report(train_metrics, val_metrics)
        recorder.summary()

        print('Completed!', flush=True)
