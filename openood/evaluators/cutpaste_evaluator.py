import torch
from sklearn.covariance import LedoitWolf as LW
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from openood.datasets import get_dataloader
from openood.postprocessors import BasePostprocessor
from openood.utils import Config


def to_np(x):
    return x.data.cuda().numpy()


class CutPasteEvaluator:
    def __init__(self, config: Config):
        self.config = config

    def eval_ood(self,
                 net,
                 id_data_loader,
                 ood_loader_dict,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()

        # loss_avg = 0.0
        correct = 0

        embeds = []
        labels = []
        data_loader = ood_loader_dict['val']
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = torch.cat(batch['data'], 0)
                data = data.cuda()
                y = torch.arange(2)
                y = y.repeat_interleave(len(batch['data'][0]))
                labels.append(y)
                y = y.cuda()

                # forward
                embed, output = net(data)
                embeds.append(embed.cuda())

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(y.data).sum().item()

        labels = torch.cat(labels)
        embeds = torch.cat(embeds)
        embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)

        train_embeds = get_train_embeds(net, self.config)
        train_embeds = torch.nn.functional.normalize(train_embeds, p=2, dim=1)
        density = GaussianDensityTorch()
        density.fit(train_embeds)
        distances = density.predict(embeds)

        fpr, tpr, _ = roc_curve(labels, distances.cpu())
        cp_auc = auc(fpr, tpr)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['auc'] = cp_auc
        return metrics


class Density(object):
    def fit(self, embeddings):
        raise NotImplementedError

    def predict(self, embeddings):
        raise NotImplementedError


class GaussianDensityTorch(object):
    def fit(self, embeddings):
        self.mean = torch.mean(embeddings, axis=0)
        self.inv_cov = torch.Tensor(LW().fit(embeddings.cpu()).precision_,
                                    device='cpu')

    def predict(self, embeddings):
        distances = self.mahalanobis_distance(embeddings, self.mean,
                                              self.inv_cov)
        return distances

    @staticmethod
    def mahalanobis_distance(values: torch.Tensor, mean: torch.Tensor,
                             inv_covariance: torch.Tensor) -> torch.Tensor:

        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        inv_covariance = inv_covariance.cuda()
        dist = torch.einsum('im,mn,in->i', x_mu, inv_covariance, x_mu)

        return dist.sqrt()


def get_train_embeds(net, config):

    preprocessor = None
    loader_dict = get_dataloader(config.dataset, preprocessor)
    train_loader = loader_dict['train']

    train_embed = []
    train_dataiter = iter(train_loader)
    with torch.no_grad():
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Train embeds:'):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            embed, logit = net(data)
            train_embed.append(embed.cuda())

    train_embed = torch.cat(train_embed)

    return train_embed
