from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from openood.evaluators.base_evaluator import BaseEvaluator
from openood.postprocessors import BasePostprocessor
from openood.utils import Config


class OpenGanEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        self.config = config

    def eval_acc(self,
                 net,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        return super().eval_acc(net=net['backbone'],
                                data_loader=data_loader,
                                postprocessor=postprocessor,
                                epoch_idx=epoch_idx)

    def eval_ood(self, net, id_data_loader: DataLoader,
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor):
        with torch.no_grad():
            conf = self.run_test_on_closeset(id_data_loader['test'],
                                             net['backbone'], net['netD'])
            for key, value in zip(ood_data_loaders.keys(),
                                  ood_data_loaders.values()):
                if key == 'val':
                    continue
                print(u'\n' + '-' * 25 +
                      'Start OpenGan evaluation on {} category'.format(key) +
                      '-' * 25,
                      flush=True)
                ood_data_loaders = value
                self.eval_auroc(net['backbone'], net['netD'],
                                id_data_loader['test'], ood_data_loaders, conf)

    def eval_auroc(self, extractor, discriminator, closedset_data,
                   openset_data, conf):
        try:
            openset_data.__delitem__(self.config.dataset.name)
        except Exception:
            pass

        extractor.eval()

        # conf = self.run_test_on_closeset(closedset_data,
        #                                  extractor, discriminator)
        self.run_test_on_openset(openset_data, extractor, discriminator, conf)

    def run_test_on_closeset(self,
                             closeset_dataloader,
                             extractor,
                             discriminator,
                             device='cuda:0'):
        feat_closedset = torch.tensor([]).type(torch.float)
        label_closedset = torch.tensor([]).type(torch.float)
        conf_closedset = torch.tensor([]).type(torch.float)

        for sample in closeset_dataloader:
            image = sample['data']
            label = sample['label']
            image = image.to(device)
            label = label.type(torch.long).view(-1).to(device)
            _, feats = extractor(image, return_feature=True)

            feats = feats.unsqueeze_(-1).unsqueeze_(-1)
            predConf = discriminator(feats)
            predConf = predConf.view(-1, 1)
            conf_closedset = torch.cat(
                (conf_closedset, predConf.reshape(-1).detach().cpu()), 0)
            import pdb
            pdb.set_trace()
            feats = feats.squeeze()
            feat_closedset = torch.cat((feat_closedset, feats.detach().cpu()))
            label_closedset = torch.cat(
                (label_closedset,
                 label.type(torch.float).detach().cpu().reshape(-1, 1)))

        conf_closedset = conf_closedset.detach().cpu().numpy()
        print(conf_closedset.shape)

        return conf_closedset

    def run_test_on_openset(self,
                            ood_loader_dict,
                            extractor,
                            discriminator,
                            conf,
                            device='cuda:0'):
        for key, dataloader in zip(ood_loader_dict.keys(),
                                   ood_loader_dict.values()):
            print('Start testing on openset -> {}: '.format(key))

            feat_openset = torch.tensor([]).type(torch.float)
            label_openset = torch.tensor([]).type(torch.float)
            conf_openset = torch.tensor([]).type(torch.float)
            for sample in dataloader:
                image = sample['data']
                label = sample['label']
                image = image.to(device)
                _, feats = extractor(image, return_feature=True)
                feats = feats.unsqueeze_(-1).unsqueeze_(-1)
                predConf = discriminator(feats)
                predConf = predConf.view(-1, 1).detach()
                conf_openset = torch.cat(
                    (conf_openset, predConf.reshape(-1).detach().cpu()), 0)

                feats = feats.squeeze()
                feat_openset = torch.cat((feat_openset, feats.detach().cpu()))
                label_openset = torch.cat(
                    (label_openset,
                     label.type(torch.float).detach().cpu().reshape(-1, 1)))

            conf_openset = conf_openset.detach().cpu().numpy()
            # roc_score, roc_to_plot = self.evaluate_openset(-conf,
            #                                                -conf_openset)

            y_true = np.array([0] * len(-conf) + [1] * len(-conf_openset))
            y_discriminator = np.concatenate([-conf, -conf_openset])
            roc_score = roc_auc_score(y_true, y_discriminator)

            print(conf.shape, conf_openset.shape)
            # plt.plot(roc_to_plot['fp'], roc_to_plot['tp'])
            # plt.grid('on')
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('ROC score {:.5f}'.format(roc_score))
            print(roc_score)

    def evaluate_openset(self, scores_closeset, scores_openset):
        y_true = np.array([0] * len(scores_closeset) +
                          [1] * len(scores_openset))
        y_discriminator = np.concatenate([scores_closeset, scores_openset])
        auc_d, roc_to_plot = self.plot_roc(y_true, y_discriminator,
                                           'Discriminator ROC')
        return auc_d, roc_to_plot

    def plot_roc(self,
                 y_true,
                 y_score,
                 title='Receiver Operating Characteristic',
                 **options):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        roc_to_plot = {
            'tp': tpr,
            'fp': fpr,
            'thresh': thresholds,
            'auc_score': auc_score
        }
        # plot = plot_xy(fpr, tpr, x_axis="False Positive Rate",
        #                y_axis="True Positive Rate", title=title)
        # if options.get('roc_output'):
        #     print("Saving ROC scores to file")
        #     np.save(options['roc_output'], (fpr, tpr))
        # return auc_score, plot, roc_to_plot
        return auc_score, roc_to_plot
