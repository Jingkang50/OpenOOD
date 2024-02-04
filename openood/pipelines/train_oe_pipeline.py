import numpy as np
import torch

import openood.utils.comm as comm
from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger


class TrainOEPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        train_loader, val_loader = loader_dict['train'], loader_dict['val']
        train_oe_loader = loader_dict['oe']
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)
        if self.config.num_gpus * self.config.num_machines > 1:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

        # init trainer and evaluator
        trainer = get_trainer(net, [train_loader, train_oe_loader], None,
                              self.config)
        evaluator = get_evaluator(self.config)

        if comm.is_main_process():
            # init recorder
            recorder = get_recorder(self.config)
            print('Start training...', flush=True)

        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            if isinstance(train_loader.sampler,
                          torch.utils.data.distributed.DistributedSampler):
                train_loader.sampler.set_epoch(epoch_idx - 1)

            # train and eval the model
            net, train_metrics = trainer.train_epoch(epoch_idx)
            val_metrics = evaluator.eval_acc(net, val_loader, None, epoch_idx)
            comm.synchronize()
            if comm.is_main_process():
                # save model and report the result
                recorder.save_model(net, val_metrics)
                recorder.report(train_metrics, val_metrics)

        if comm.is_main_process():
            recorder.summary()
            print(u'\u2500' * 70, flush=True)

            # evaluate on test set
            print('Start testing...', flush=True)

        test_metrics = evaluator.eval_acc(net, test_loader)

        if comm.is_main_process():
            print('\nComplete Evaluation, Last accuracy {:.2f}'.format(
                100.0 * test_metrics['acc']),
                  flush=True)
            print('Completed!', flush=True)
