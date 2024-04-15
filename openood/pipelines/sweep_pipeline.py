from copy import deepcopy
import wandb
from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger

import torch.nn as nn

from openood.utils.config import Config, merge_configs

def setup_wandb_sweep(config):
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'optimizer.lr': {'min': 0.00001, 'max': 0.001},
            'optimizer.batch_size': {'values': [4, 8, 16]},
            'optimizer.name': {'values': ['adam', 'sgd']},
            'optimizer.num_epochs': {'values': [5,]},
            'optimizer.momentum': {'values': [0.9, 0.95]},
            'optimizer.weight_decay': {'values': [0.0001, 0.001]},
            'optimizer.beta1': {'values': [0.9, 0.95]},  # specific to Adam
            'optimizer.beta2': {'values': [0.999]},  # specific to Adam
            'optimizer.epsilon': {'values': [1e-7, 1e-8]},  # specific to Adam
            'optimizer.nesterov': {'values': [True, False]}  # specific to SGD
        }
    }
    return wandb.sweep(sweep_config, project=config.exp_name)

class SweepPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def _run(self):
        """This method runs the normal training and evaluation pipeline."""
        wandb.init(project=self.config, config=self.config)

        config = deepcopy(self.config)

        # Sync W&B config with the local config if there are overrides from W&B
        # for key, value in wandb.config.items():
        #     setattr(config, key, value)
        config = merge_configs(config, Config(wandb.config.as_dict()))

        # Generate output directory and save the full config file
        setup_logger(config)

        # Get dataloader
        loader_dict = get_dataloader(config)
        train_loader, val_loader = loader_dict['train'], loader_dict['val']
        test_loader = loader_dict.get('test')

        # Init network
        net = get_network(config.network)
        assert isinstance(net, nn.Module), 'Network must be a torch.nn.Module.'

        # Init trainer with W&B integration
        trainer = get_trainer(net, train_loader, config=config)

        # Init evaluator and recorder
        evaluator = get_evaluator(config)
        recorder = get_recorder(config)

        # Training and evaluation loop
        for epoch_idx in range(1, config.optimizer.num_epochs + 1):
            train_metrics = trainer.train_epoch(epoch_idx)
            val_metrics = evaluator.eval_acc(net, val_loader, None, epoch_idx)
            recorder.save_model(net, val_metrics)
            recorder.report(train_metrics, val_metrics)
        recorder.summary()

        # Evaluate on test set
        if test_loader is not None:
            test_metrics = evaluator.eval_acc(net, test_loader)
            wandb.log({'final_test_accuracy': test_metrics['acc']})
        
        wandb.finish()

    def run(self):
        """This method initializes the W&B sweep and assigns _run as the target function."""
        sweep_id = setup_wandb_sweep(self.config)  # Ensure this function is defined to setup and return a sweep ID
        wandb.agent(sweep_id, function=self._run)

# Usage of this class would be:
# config = Config()  # Define how you obtain your configuration
# pipeline = FinetunePipeline(config)
# pipeline.run()  # To run with a sweep
