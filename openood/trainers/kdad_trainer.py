import torch
from torch.autograd import Variable
from tqdm import tqdm

from openood.losses.kdad_losses import DirectionOnlyLoss, MseDirectionLoss
from openood.utils import Config


class KdadTrainer:
    def __init__(self, net, train_loader, config: Config):
        self.vgg = net['vgg']
        self.model = net['model']
        self.train_loader = train_loader
        self.config = config
        # choose loss type
        if self.config['direction_loss_only']:
            self.criterion = DirectionOnlyLoss()
        else:
            self.criterion = MseDirectionLoss(self.config['lamda'])
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=float(
                                              self.config['learning_rate']))

    def train_epoch(self, epoch_idx):

        self.model.train()
        epoch_loss = 0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}'.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            X = batch['data']
            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            X = Variable(X).cuda()

            # compute respective output
            output_pred = self.model.forward(X)
            output_real = self.vgg(X)

            # compute loss
            total_loss = self.criterion(output_pred, output_real)

            # Add loss to the list
            epoch_loss += total_loss.item()

            # Clear the previous gradients
            self.optimizer.zero_grad()

            # Compute gradients
            total_loss.backward()

            # Adjust weights
            self.optimizer.step()
        net = {}
        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['epoch_loss'] = epoch_loss
        net['vgg'] = self.vgg
        net['model'] = self.model
        return net, metrics
