from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger


class TrainARPLGANPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        train_loader, val_loader = loader_dict['train'], loader_dict['val']
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, self.config)
        self.config.trainer.name = 'arpl'
        trainer_aux = get_trainer(net, train_loader, self.config)
        evaluator = get_evaluator(self.config)

        # init recorder
        recorder = get_recorder(self.config)

        print('Start training...', flush=True)
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            # train and eval the model
            net, train_metrics = trainer.train_epoch(epoch_idx)
            net, train_aux_metrics = trainer_aux.train_epoch(epoch_idx)
            train_metrics['loss'] = train_aux_metrics['loss']
            val_metrics = evaluator.eval_acc(net, val_loader, None, epoch_idx)
            trainer.scheduler.step()
            # save model and report the result
            recorder.save_model(net, val_metrics)
            recorder.report(train_metrics, val_metrics)
        recorder.summary()
        print(u'\u2500' * 70, flush=True)

        # evaluate on test set
        print('Start testing...', flush=True)
        test_metrics = evaluator.eval_acc(net, trainer.criterion, test_loader)
        print('\nComplete Evaluation, Last accuracy {:.2f}'.format(
            100.0 * test_metrics['acc']),
              flush=True)
        print('Completed!', flush=True)
