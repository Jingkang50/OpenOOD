from typing import Any

from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from .base_postprocessor import BasePostprocessor


class fDBDPostprocessor(BasePostprocessor):
    """Empirically, the feature norm (torch.norm(feature, dim=1)) is sometimes
    a more effective regularizer than.

    the feature distance to the training feature mean (torch.norm(feature - self.train_mean, dim=1)). In this
    implementation, we diverge slightly from the original paper by selecting the regularizer based on the
    validation set.
    """
    def __init__(self, config):
        super(fDBDPostprocessor, self).__init__(config)
        self.APS_mode = True
        self.setup_flag = False
        self.train_mean = None
        self.denominator_matrix = None
        self.num_classes = None
        self.activation_log = None

        self.args = self.config.postprocessor.postprocessor_args
        self.distance_as_normalizer = self.args.distance_as_normalizer
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # collect training mean
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)

                    activation_log.append(feature.data.cpu().numpy())

            activation_log_concat = np.concatenate(activation_log, axis=0)
            self.activation_log = activation_log_concat
            self.train_mean = torch.from_numpy(
                np.mean(activation_log_concat, axis=0)).cuda()

            # compute denominator matrix
            for i, param in enumerate(net.fc.parameters()):
                if i == 0:
                    w = param.data.cpu().numpy()
                else:
                    b = param.data.cpu().numpy()

            self.num_classes = b.shape[0]

            denominator_matrix = np.zeros((self.num_classes, self.num_classes))
            for p in range(self.num_classes):
                w_p = w - w[p, :]
                denominator = np.linalg.norm(w_p, axis=1)
                denominator[p] = 1
                denominator_matrix[p, :] = denominator

            self.denominator_matrix = torch.tensor(denominator_matrix).cuda()
            self.setup_flag = True

        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        values, nn_idx = output.max(1)
        logits_sub = torch.abs(output - values.repeat(self.num_classes, 1).T)
        if self.distance_as_normalizer:
            score = torch.sum(logits_sub / self.denominator_matrix[nn_idx],
                              axis=1) / torch.norm(feature - self.train_mean,
                                                   dim=1)
        else:
            score = torch.sum(logits_sub / self.denominator_matrix[nn_idx],
                              axis=1) / torch.norm(feature, dim=1)
        return nn_idx, score

    def set_hyperparam(self, hyperparam: list):
        self.distance_as_normalizer = hyperparam[0]

    def get_hyperparam(self):
        return self.distance_as_normalizer
