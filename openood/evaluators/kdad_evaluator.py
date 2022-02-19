from copy import deepcopy

import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import auc, roc_curve
from torch import nn
from torch.autograd import Variable
from torch.nn import ReLU
from tqdm import tqdm

from openood.postprocessors.kdad_postprocessor import (convert_to_grayscale,
                                                       max_regarding_to_abs,
                                                       morphological_process)


class KdadDetectionEvaluator:
    """It is the kdad evaluator for Anomaly Detection."""
    def __init__(self, config):
        self.config = config

    def eval_ood(self,
                 net,
                 id_loader_dict,
                 ood_loader_dict,
                 postprocessor,
                 epoch_idx=-1):
        config = self.config
        id_loader = id_loader_dict['train']
        ood_loader = ood_loader_dict['val']
        normal_class = config['normal_class']
        lamda = config['lamda']
        dataset_name = config['dataset_name']
        direction_only = config['direction_loss_only']
        model = net['model']
        vgg = net['vgg']

        # set target_class
        if dataset_name != 'mvtec':
            target_class = normal_class
        else:
            mvtec_good_dict = {
                'bottle': 3,
                'cable': 5,
                'capsule': 2,
                'carpet': 2,
                'grid': 3,
                'hazelnut': 2,
                'leather': 4,
                'metal_nut': 3,
                'pill': 5,
                'screw': 0,
                'tile': 2,
                'toothbrush': 1,
                'transistor': 3,
                'wood': 2,
                'zipper': 4
            }
            target_class = mvtec_good_dict[normal_class]

        similarity_loss = torch.nn.CosineSimilarity()
        label_score_id = []
        label_score_ood = []
        model.eval()
        eval_dataiter_id = iter(id_loader)
        eval_dataiter_ood = iter(ood_loader)

        # start evaluation
        for train_step in tqdm(range(1,
                                     len(eval_dataiter_id) + 1),
                               desc='id',
                               position=0,
                               leave=True):
            batch = next(eval_dataiter_id)
            X, Y = batch['data'], batch['label']

            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            X = Variable(X).cuda()
            output_pred = model.forward(X)
            output_real = vgg(X)
            y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[
                9], output_pred[12]
            y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]

            if direction_only:
                loss_1 = 1 - similarity_loss(
                    y_pred_1.view(y_pred_1.shape[0], -1),
                    y_1.view(y_1.shape[0], -1))
                loss_2 = 1 - similarity_loss(
                    y_pred_2.view(y_pred_2.shape[0], -1),
                    y_2.view(y_2.shape[0], -1))
                loss_3 = 1 - similarity_loss(
                    y_pred_3.view(y_pred_3.shape[0], -1),
                    y_3.view(y_3.shape[0], -1))
                total_loss = loss_1 + loss_2 + loss_3
            else:
                abs_loss_1 = torch.mean((y_pred_1 - y_1)**2, dim=(1, 2, 3))
                loss_1 = 1 - similarity_loss(
                    y_pred_1.view(y_pred_1.shape[0], -1),
                    y_1.view(y_1.shape[0], -1))
                abs_loss_2 = torch.mean((y_pred_2 - y_2)**2, dim=(1, 2, 3))
                loss_2 = 1 - similarity_loss(
                    y_pred_2.view(y_pred_2.shape[0], -1),
                    y_2.view(y_2.shape[0], -1))
                abs_loss_3 = torch.mean((y_pred_3 - y_3)**2, dim=(1, 2, 3))
                loss_3 = 1 - similarity_loss(
                    y_pred_3.view(y_pred_3.shape[0], -1),
                    y_3.view(y_3.shape[0], -1))
                total_loss = loss_1 + loss_2 + loss_3 + lamda * (
                    abs_loss_1 + abs_loss_2 + abs_loss_3)

            label_score_id += list(
                zip(Y.cpu().data.numpy().tolist(),
                    total_loss.cpu().data.numpy().tolist()))
        for train_step in tqdm(range(1,
                                     len(eval_dataiter_ood) + 1),
                               desc='ood',
                               position=0,
                               leave=True):
            batch = next(eval_dataiter_ood)
            X, Y = batch['data'], batch['label']

            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            X = Variable(X).cuda()
            output_pred = model.forward(X)
            output_real = vgg(X)
            y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[
                9], output_pred[12]
            y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]

            if direction_only:
                loss_1 = 1 - similarity_loss(
                    y_pred_1.view(y_pred_1.shape[0], -1),
                    y_1.view(y_1.shape[0], -1))
                loss_2 = 1 - similarity_loss(
                    y_pred_2.view(y_pred_2.shape[0], -1),
                    y_2.view(y_2.shape[0], -1))
                loss_3 = 1 - similarity_loss(
                    y_pred_3.view(y_pred_3.shape[0], -1),
                    y_3.view(y_3.shape[0], -1))
                total_loss = loss_1 + loss_2 + loss_3
            else:
                abs_loss_1 = torch.mean((y_pred_1 - y_1)**2, dim=(1, 2, 3))
                loss_1 = 1 - similarity_loss(
                    y_pred_1.view(y_pred_1.shape[0], -1),
                    y_1.view(y_1.shape[0], -1))
                abs_loss_2 = torch.mean((y_pred_2 - y_2)**2, dim=(1, 2, 3))
                loss_2 = 1 - similarity_loss(
                    y_pred_2.view(y_pred_2.shape[0], -1),
                    y_2.view(y_2.shape[0], -1))
                abs_loss_3 = torch.mean((y_pred_3 - y_3)**2, dim=(1, 2, 3))
                loss_3 = 1 - similarity_loss(
                    y_pred_3.view(y_pred_3.shape[0], -1),
                    y_3.view(y_3.shape[0], -1))
                total_loss = loss_1 + loss_2 + loss_3 + lamda * (
                    abs_loss_1 + abs_loss_2 + abs_loss_3)

            label_score_ood += list(
                zip(Y.cpu().data.numpy().tolist(),
                    total_loss.cpu().data.numpy().tolist()))

        # compute roc_auc
        labels, scores = zip(*(label_score_id + label_score_ood))
        labels = np.array(labels)
        indx1 = labels == target_class
        indx2 = labels != target_class
        labels[indx1] = 1
        labels[indx2] = 0
        scores = np.array(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
        metrics = {}
        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc, 4)
        metrics['epoch_idx'] = epoch_idx
        metrics['roc_auc'] = roc_auc

        return metrics


# require to amend
class KdadLocalizationEvaluator:
    def __init__(self, config):
        self.config = config

    def eval_acc(self, model, vgg, test_dataloader, ground_truth):
        config = self.config
        localization_method = config['localization_method']
        if localization_method == 'gradients':
            grad = gradients_localization(model, vgg, test_dataloader, config)
        if localization_method == 'smooth_grad':
            grad = smooth_grad_localization(model, vgg, test_dataloader,
                                            config)
        if localization_method == 'gbp':
            grad = gbp_localization(model, vgg, test_dataloader, config)

        return compute_localization_auc(grad, ground_truth)


class VanillaSaliency():
    def __init__(self, model, vgg, device, config):
        self.model = model
        self.vgg = vgg
        self.device = device
        self.config = config
        self.model.eval()

    def generate_saliency(self, data, make_single_channel=True):
        data_var_sal = Variable(data).to(self.device)
        self.model.zero_grad()
        if data_var_sal.grad is not None:
            data_var_sal.grad.data.zero_()
        data_var_sal.requires_grad_(True)

        lamda = self.config['lamda']
        criterion = nn.MSELoss()
        similarity_loss = torch.nn.CosineSimilarity()

        output_pred = self.model.forward(data_var_sal)
        output_real = self.vgg(data_var_sal)
        y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[
            9], output_pred[12]
        y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]

        abs_loss_1 = criterion(y_pred_1, y_1)
        loss_1 = torch.mean(1 - similarity_loss(
            y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
        abs_loss_2 = criterion(y_pred_2, y_2)
        loss_2 = torch.mean(1 - similarity_loss(
            y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
        abs_loss_3 = criterion(y_pred_3, y_3)
        loss_3 = torch.mean(1 - similarity_loss(
            y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))
        total_loss = loss_1 + loss_2 + loss_3 + lamda * (
            abs_loss_1 + abs_loss_2 + abs_loss_3)
        self.model.zero_grad()
        total_loss.backward()
        grad = data_var_sal.grad.data.detach().cpu()

        if make_single_channel:
            grad = np.asarray(grad.detach().cpu().squeeze(0))
            grad = convert_to_grayscale(grad)
        else:
            grad = np.asarray(grad)
        return grad


def grad_calc(inputs, model, vgg, config):
    inputs = inputs.cuda()
    inputs.requires_grad = True
    temp = torch.zeros(inputs.shape)
    lamda = config['lamda']
    criterion = nn.MSELoss()
    similarity_loss = torch.nn.CosineSimilarity()

    for i in range(inputs.shape[0]):
        output_pred = model.forward(inputs[i].unsqueeze(0), target_layer=14)
        output_real = vgg(inputs[i].unsqueeze(0))
        y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[
            9], output_pred[12]
        y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]
        abs_loss_1 = criterion(y_pred_1, y_1)
        loss_1 = torch.mean(1 - similarity_loss(
            y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
        abs_loss_2 = criterion(y_pred_2, y_2)
        loss_2 = torch.mean(1 - similarity_loss(
            y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
        abs_loss_3 = criterion(y_pred_3, y_3)
        loss_3 = torch.mean(1 - similarity_loss(
            y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))
        total_loss = loss_1 + loss_2 + loss_3 + lamda * (
            abs_loss_1 + abs_loss_2 + abs_loss_3)
        model.zero_grad()
        total_loss.backward()

        temp[i] = inputs.grad[i]

    return temp


class GuidedBackprop():
    def __init__(self, model, vgg, device):
        self.model = model
        self.vgg = vgg
        self.gradients = None
        self.forward_relu_outputs = []
        self.device = device
        self.hooks = []
        self.model.eval()
        self.update_relus()

    def update_relus(self):
        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(
                grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out, )

        def relu_forward_hook_function(module, ten_in, ten_out):
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for module in self.model.modules():
            if isinstance(module, ReLU):
                self.hooks.append(
                    module.register_backward_hook(relu_backward_hook_function))
                self.hooks.append(
                    module.register_forward_hook(relu_forward_hook_function))

    def generate_gradients(self,
                           input_image,
                           config,
                           make_single_channel=True):
        vanillaSaliency = VanillaSaliency(self.model,
                                          self.vgg,
                                          self.device,
                                          config=config)
        sal = vanillaSaliency.generate_saliency(input_image,
                                                make_single_channel)
        if not isinstance(sal, np.ndarray):
            sal = sal.detach().cpu().numpy()
        for hook in self.hooks:
            hook.remove()
        return sal


def gradients_localization(model, vgg, test_dataloader, config):
    model.eval()
    print('Vanilla Backpropagation:')
    temp = None
    for data in test_dataloader:
        X, Y = data
        grad = grad_calc(X, model, vgg, config)
        temp = np.zeros((grad.shape[0], grad.shape[2], grad.shape[3]))
        for i in range(grad.shape[0]):
            grad_temp = convert_to_grayscale(grad[i].cpu().numpy())
            grad_temp = grad_temp.squeeze(0)
            grad_temp = gaussian_filter(grad_temp, sigma=4)
            temp[i] = grad_temp
    return temp


def smooth_grad_localization(model, vgg, test_dataloader, config):
    model.eval()
    print('Smooth Grad Method:')

    grad1 = None
    i = 0

    for data in test_dataloader:
        X, Y = data
        grad1 = np.zeros((X.shape[0], 1, 128, 128), dtype=np.float32)
        for x in X:
            data = x.view(1, 3, 128, 128)

            vbp = VanillaSaliency(model, vgg, 'cuda:0', config)

            smooth_grad_saliency = abs(
                generate_smooth_grad(data, 50, 0.05, vbp))
            smooth_grad_saliency = (smooth_grad_saliency -
                                    min(smooth_grad_saliency.flatten())) / (
                                        max(smooth_grad_saliency.flatten()) -
                                        min(smooth_grad_saliency.flatten()))
            saliency = smooth_grad_saliency

            saliency = gaussian_filter(saliency, sigma=4)
            grad1[i] = saliency
            i += 1

    grad1 = grad1.reshape(-1, 128, 128)
    return grad1


def compute_localization_auc(grad, x_ground):
    tpr = []
    fpr = []
    x_ground_comp = np.mean(x_ground, axis=3)

    thresholds = [0.001 * i for i in range(1000)]

    for threshold in thresholds:
        grad_t = 1.0 * (grad >= threshold)
        grad_t = morphological_process(grad_t)
        tp_map = np.multiply(grad_t, x_ground_comp)
        tpr.append(np.sum(tp_map) / np.sum(x_ground_comp))

        inv_x_ground = 1 - x_ground_comp
        fp_map = np.multiply(grad_t, inv_x_ground)
        tn_map = np.multiply(1 - grad_t, 1 - x_ground_comp)
        fpr.append(np.sum(fp_map) / (np.sum(fp_map) + np.sum(tn_map)))

    return auc(fpr, tpr)


def gbp_localization(model, vgg, test_dataloader, config):
    model.eval()
    print('GBP Method:')

    grad1 = None
    i = 0

    for data in test_dataloader:
        X, Y = data
        grad1 = np.zeros((X.shape[0], 1, 128, 128), dtype=np.float32)
        for x in X:
            data = x.view(1, 3, 128, 128)

            GBP = GuidedBackprop(model, vgg, 'cuda:0')
            gbp_saliency = abs(GBP.generate_gradients(data, config))
            gbp_saliency = (gbp_saliency - min(gbp_saliency.flatten())) / (
                max(gbp_saliency.flatten()) - min(gbp_saliency.flatten()))
            saliency = gbp_saliency

            saliency = gaussian_filter(saliency, sigma=4)
            grad1[i] = saliency
            i += 1

    grad1 = grad1.reshape(-1, 128, 128)
    return grad1


def generate_smooth_grad(data,
                         param_n,
                         param_sigma_multiplier,
                         vbp,
                         single_channel=True):
    smooth_grad = None

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(data) - torch.min(data)).item()
    VBP = vbp
    for x in range(param_n):
        noise = Variable(data.data.new(data.size()).normal_(mean, sigma**2))
        noisy_img = data + noise
        vanilla_grads = VBP.generate_saliency(noisy_img, single_channel)
        if not isinstance(vanilla_grads, np.ndarray):
            vanilla_grads = vanilla_grads.detach().cpu().numpy()
        if smooth_grad is None:
            smooth_grad = vanilla_grads
        else:
            smooth_grad = smooth_grad + vanilla_grads

    smooth_grad = smooth_grad / param_n
    return smooth_grad


class IntegratedGradients():
    def __init__(self, model, vgg, device):
        self.model = model
        self.vgg = vgg
        self.gradients = None
        self.device = device
        # Put model in evaluation mode
        self.model.eval()

    def generate_images_on_linear_path(self, input_image, steps):
        step_list = np.arange(steps + 1) / steps
        xbar_list = [input_image * step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, make_single_channel=True):
        vanillaSaliency = VanillaSaliency(self.model, self.vgg, self.device)
        saliency = vanillaSaliency.generate_saliency(input_image,
                                                     make_single_channel)
        if not isinstance(saliency, np.ndarray):
            saliency = saliency.detach().cpu().numpy()
        return saliency

    def generate_integrated_gradients(self,
                                      input_image,
                                      steps,
                                      make_single_channel=True):
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        integrated_grads = None
        for xbar_image in xbar_list:
            single_integrated_grad = self.generate_gradients(xbar_image, False)
            if integrated_grads is None:
                integrated_grads = deepcopy(single_integrated_grad)
            else:
                integrated_grads = (integrated_grads + single_integrated_grad)
        integrated_grads /= steps
        saliency = integrated_grads[0]
        img = input_image.detach().cpu().numpy().squeeze(0)
        saliency = np.asarray(saliency) * img
        if make_single_channel:
            saliency = max_regarding_to_abs(np.max(saliency, axis=0),
                                            np.min(saliency, axis=0))
        return saliency
