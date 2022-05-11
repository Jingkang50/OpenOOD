import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from openood.utils import Config


class Conf_Esti_Trainer:
    def __init__(self, net, train_loader, config: Config) -> None:
        self.train_loader = train_loader
        self.config = config
        self.net = net
        self.prediction_criterion = nn.NLLLoss().cuda()
        self.optimizer = torch.optim.SGD(
            net.parameters(),
            lr=config.optimizer['learning_rate'],
            momentum=config.optimizer['momentum'],
            nesterov=config.optimizer['nesterov'],
            weight_decay=config.optimizer['weight_decay'])
        self.scheduler = MultiStepLR(self.optimizer,
                                     milestones=config.scheduler['milestones'],
                                     gamma=config.scheduler['gamma'])
        self.lmbda = self.config.trainer['lmbda']

    def train_epoch(self, epoch_idx):
        self.net.train()
        correct_count = 0.
        total = 0.
        accuracy = 0.
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}'.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            images = Variable(batch['data']).cuda()
            labels = Variable(batch['label']).cuda()
            labels_onehot = Variable(
                encode_onehot(labels, self.config.num_classes))
            self.net.zero_grad()
            pred_original, confidence = self.net(images)
            pred_original = F.softmax(pred_original, dim=-1)
            confidence = torch.sigmoid(confidence)
            eps = self.config.trainer['eps']
            pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
            confidence = torch.clamp(confidence, 0. + eps, 1. - eps)

            if not self.config.baseline:
                # Randomly set half of the confidences to 1 (i.e. no hints)
                b = Variable(
                    torch.bernoulli(
                        torch.Tensor(confidence.size()).uniform_(0,
                                                                 1))).cuda()
                conf = confidence * b + (1 - b)
                pred_new = pred_original * conf.expand_as(
                    pred_original) + labels_onehot * (
                        1 - conf.expand_as(labels_onehot))
                pred_new = torch.log(pred_new)
            else:
                pred_new = torch.log(pred_original)

            xentropy_loss = self.prediction_criterion(pred_new, labels)
            confidence_loss = torch.mean(-torch.log(confidence))

            if self.config.baseline:
                total_loss = xentropy_loss
            else:
                total_loss = xentropy_loss + (self.lmbda * confidence_loss)

                if self.config.trainer['budget'] > confidence_loss.item():
                    self.lmbda = self.lmbda / 1.01
                elif self.config.trainer['budget'] <= confidence_loss.item():
                    self.lmbda = self.lmbda / 0.99

            total_loss.backward()
            self.optimizer.step()

            pred_idx = torch.max(pred_original.data, 1)[1]
            total += labels.size(0)
            correct_count += (pred_idx == labels.data).sum()
            accuracy = correct_count / total
        self.scheduler.step()
        metrics = {}
        metrics['train_acc'] = accuracy
        metrics['epoch_idx'] = epoch_idx
        return self.net, metrics


def encode_onehot(labels, n_classes):
    onehot = torch.FloatTensor(labels.size()[0],
                               n_classes)  # batchsize * num of class
    labels = labels.data
    if labels.is_cuda:
        onehot = onehot.cuda()
    onehot.zero_()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot


"""
[0,0,0,1,0,0,0,0,0,0]
[0,0,0,0,0,0,0,1,0,0]
[0,0,0,0,0,1,0,0,0,0]
...

"""
