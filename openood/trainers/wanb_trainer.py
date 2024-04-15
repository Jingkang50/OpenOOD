import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.utils import Config
from .lr_scheduler import cosine_annealing

class WandBTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader, config: Config) -> None:
        self.net = net
        self.train_loader = train_loader
        self.config = config

        self.setup_training_components(config.optimizer)

    def setup_training_components(self, optmizer_config: Config):
        # Select optimizer based on the W&B config
        if optmizer_config.name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=optmizer_config.lr,
                betas=(optmizer_config.beta1, optmizer_config.beta2),
                eps=optmizer_config.epsilon,
                weight_decay=optmizer_config.weight_decay
            )
        elif optmizer_config.name == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=optmizer_config.lr,
                momentum=optmizer_config.momentum,
                weight_decay=optmizer_config.weight_decay,
                nesterov=optmizer_config.nesterov
            )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                optmizer_config.num_epochs * len(self.train_loader),
                1,
                1e-6 / optmizer_config.lr,
            )
        )

    def train_epoch(self, epoch_idx) -> dict[str, float]:
        self.net.train()
        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(len(train_dataiter)),
                               desc=f'Epoch {epoch_idx:03d}: ',
                               position=0, leave=True):
            # Prepare the data
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()
            
            # Zero the gradients
            self.optimizer.zero_grad()

            # Make a forward pass
            logits = self.net(data)

            # Compute the loss and backpropagate
            loss = F.cross_entropy(logits, target)
            loss.backward()

            # backward
            self.optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        
            wandb.log({"loss": loss_avg, "epoch": epoch_idx})

        return {"loss": loss_avg, "epoch_idx": epoch_idx}

# def train():
#     with wandb.init() as run:
#         config = Config()  # Your method to create a Config object
#         net = get_network(config.network)
#         train_loader = get_dataloader(config)["train"]
#         trainer = WandBTrainer(net, train_loader, config)
#         for epoch in range(1, run.config.epochs + 1):
#             loss = trainer.train_epoch(epoch)
#             wandb.log({"val_loss": loss, "epoch": epoch})

# sweep_id = setup_wandb_sweep()
# wandb.agent(sweep_id, train)
