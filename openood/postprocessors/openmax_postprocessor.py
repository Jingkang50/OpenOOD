import libmr
import numpy as np
import scipy.spatial.distance as spd
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class OpenMax(BasePostprocessor):
    def __init__(self, config):
        super(OpenMax, self).__init__(config)
        self.nc = config.dataset.num_classes
        self.weibull_alpha = 3
        self.weibull_threshold = 0.9
        self.weibull_tail = 20

    def setup(self, net: nn.Module, train_loader_dict, ood_loder_dict):

        # Fit the weibull distribution from training data.
        print('Fittting Weibull distribution...')
        _, mavs, dists = compute_train_score_and_mavs_and_dists(
            self.nc, train_loader_dict['train'], device='cuda', net=net)
        categories = list(range(0, self.nc))
        self.weibull_model = fit_weibull(mavs, dists, categories,
                                         self.weibull_tail, 'euclidean')

    def postprocess(self, net: nn.Module, data):
        net.eval()

        device = 'cuda'
        scores = []
        with torch.no_grad():
            for inputs in data.split(1, dim=0):
                inputs = inputs.to(device)
                outputs = net(inputs)
                # loss = criterion(outputs, targets)
                # test_loss += loss.item()
                # _, predicted = outputs.max(1)
                scores.append(outputs)

                # total += targets.size(0)
                # correct += predicted.eq(targets).sum().item()

        # Get the prdict results.
        scores = torch.cat(scores, dim=0).cpu().numpy()
        scores = np.array(scores)[:, np.newaxis, :]

        categories = list(range(0, self.nc))

        pred_openmax = []
        score_openmax = []
        score_softmax = []
        for score in scores:
            so, ss = openmax(self.weibull_model, categories, score, 0.5,
                             self.weibull_alpha,
                             'euclidean')  # openmax_prob, softmax_prob
            pred_openmax.append(
                np.argmax(so) if np.max(so) >= self.weibull_threshold else (
                    self.nc - 1))

            score_openmax.append(so)

            softmax_conf = np.max(ss)
            score_softmax.append(softmax_conf)

        pred = []
        for i in pred_openmax:
            pred.append(torch.tensor(i))

        conf = []
        for i in score_openmax:
            conf.append(i)

        conf = torch.tensor(conf, dtype=torch.float32)
        conf = conf.cuda()

        return torch.tensor(pred), torch.max(conf, dim=1)[0]  # conf[:, -1]


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV
        for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append(
            [spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([
            spd.euclidean(mcv, feat[channel]) * eu_weight +
            spd.cosine(mcv, feat[channel]) for feat in features
        ])

    return {
        'eucos': np.array(eucos_dists),
        'cosine': np.array(cos_dists),
        'euclidean': np.array(eu_dists)
    }


def compute_train_score_and_mavs_and_dists(train_class_num, trainloader,
                                           device, net):
    scores = [[] for _ in range(train_class_num)]

    train_dataiter = iter(trainloader)
    with torch.no_grad():
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Progress: ',
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # this must cause error for cifar
            outputs = net(data)
            for score, t in zip(outputs, target):

                if torch.argmax(score) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))

    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    dists = [
        compute_channel_distances(mcv, score)
        for mcv, score in zip(mavs, scores)
    ]
    return scores, mavs, dists


def fit_weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances
                        and save weibull model parameters for re-adjusting
                        softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(
            distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [
        weibull_model[category_name]['mean_vec'],
        weibull_model[category_name]['distances_{}'.format(distance_type)],
        weibull_model[category_name]['weibull_model']
    ]


def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print('distance type not known: enter either of eucos, \
               euclidean or cosine')
    return query_distance


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model,
            categories,
            input_score,
            eu_weight,
            alpha=10,
            distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)

    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha)
                     for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model,
                                             distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel],
                                         eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob
