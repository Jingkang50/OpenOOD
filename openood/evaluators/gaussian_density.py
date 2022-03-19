from sklearn.covariance import LedoitWolf as LW
import torch


class Density(object):
    def fit(self, embeddings):
        raise NotImplementedError

    def predict(self, embeddings):
        raise NotImplementedError


class GaussianDensityTorch(object):

    def fit(self, embeddings):
        self.mean = torch.mean(embeddings, axis=0)
        self.inv_cov = torch.Tensor(LW().fit(embeddings.cpu()).precision_,device="cpu")

    def predict(self, embeddings):
        distances = self.mahalanobis_distance(embeddings,
                                              self.mean, self.inv_cov)
        return distances

    @staticmethod
    def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        
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
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        
        return dist.sqrt()
