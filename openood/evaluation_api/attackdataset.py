from typing import Callable, List, Type

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as trn
from torchvision import transforms
from tqdm import tqdm
import time
from datetime import datetime

from .datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from .preprocessor import get_default_preprocessor
from .preprocessor import default_preprocessing_dict

from openood.attacks.misc import (
    create_dir,
    create_log_file,
    save_log
)

from foolbox import PyTorchModel
import foolbox.attacks as fa
from foolbox.criteria import Misclassification 

home = os.getenv('HOME')

DEEPFOOL   = ['fgsm', 'bim', 'pgd', 'df', 'cw'] + ['pgd_bpda']
AUTOATTACK = ['aa', 'apgd-ce', 'square']

class ModelWrapper:
    def __init__(self, model, mean, std):
        self.model = model
        self.normalization = transforms.Normalize(mean, std)

    def normalize_input(self, input_data):
        # Apply the stored normalization to the input data
        normalized_input = self.normalization(input_data)
        return normalized_input

    def __getattr__(self, attr):
        # Delegate attribute access to the original model
        return getattr(self.model, attr)

    def __call__(self, input_data):
        normalized_input = self.normalize_input(input_data)
        return self.model(normalized_input)


class AttackDataset:
    def __init__(
        self,
        net: nn.Module,
        id_name: str,
        data_root: str = './data',
        config_root: str = './configs',
        preprocessor: Callable = None,
        batch_size: int = 200,
        shuffle: bool = False,
        num_workers: int = 4,
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) discriminative OOD
        detection methods.

        Args:
            net (nn.Module):
                The base classifier.
            id_name (str):
                The name of the in-distribution dataset.
            data_root (str, optional):
                The path of the data folder. Defaults to './data'.
            config_root (str, optional):
                The path of the config folder. Defaults to './configs'.
            preprocessor (Callable, optional):
                The preprocessor of input images.
                Passing None will use the default preprocessor
                following convention. Defaults to None.
            postprocessor_name (str, optional):
                The name of the postprocessor that obtains OOD score.
                Ignored if an actual postprocessor is passed.
                Defaults to None.
            postprocessor (Type[BasePostprocessor], optional):
                An actual postprocessor instance which inherits
                OpenOOD's BasePostprocessor. Defaults to None.
            batch_size (int, optional):
                The batch size of samples. Defaults to 200.
            shuffle (bool, optional):
                Whether shuffling samples. Defaults to False.
            num_workers (int, optional):
                The num_workers argument that will be passed to
                data loaders. Defaults to 4.

        Raises:
            ValueError:
                If both postprocessor_name and postprocessor are None.
            ValueError:
                If the specified ID dataset {id_name} is not supported.
            TypeError:
                If the passed postprocessor does not inherit BasePostprocessor.
        """
        # # check the arguments
        if id_name not in DATA_INFO:
            raise ValueError(f'Dataset [{id_name}] is not supported')

        # get data preprocessor
        preprocessor = get_default_preprocessor(id_name)

        # set up config root
        if config_root is None:
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_root = os.path.join(*filepath.split('/')[:-2], 'configs')

        # load data
        data_setup(data_root, id_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        dataloader_dict = get_id_ood_dataloader(id_name, data_root, preprocessor, **loader_kwargs)

        preprocessor = get_default_preprocessor(id_name, att=True)

        tmp = default_preprocessing_dict[id_name]['normalization']
        normalize = {'mean':tmp[0], 'std':tmp[1]}

        self.id_name = id_name
        self.data_root = data_root
        self.net = net
        self.normalize = normalize
        self.preprocessor = preprocessor
        self.dataloader_dict = dataloader_dict
        self.metrics = {
            'id_acc': None,
            'csid_acc': None,
            'ood': None,
            'fsood': None
        }
        self.scores = {
            'id': {
                'train': None,
                'val': None,
                'test': None
            },
            'csid': {k: None
                     for k in dataloader_dict['csid'].keys()},
            'ood': {
                'val': None,
                'near':
                {k: None
                 for k in dataloader_dict['ood']['near'].keys()},
                'far': {k: None
                        for k in dataloader_dict['ood']['far'].keys()},
            },
            'id_preds': None,
            'id_labels': None,
            'csid_preds': {k: None
                           for k in dataloader_dict['csid'].keys()},
            'csid_labels': {k: None
                            for k in dataloader_dict['csid'].keys()},
        }

        self.net.eval()


    def predict(self, model, input_batch, labels):
        logits = model(input_batch)
        # Get the predicted class label
        predictions = logits.argmax(-1)

        accuracy = (predictions == labels).cpu().float().mean().item()
        print("accuracy", accuracy)
        return predictions


    def run_attack(self, args):
        
        predictions = {}
        predictions_gt  = []
        predictions_ben = []
        predictions_pgd = []

        self.net.eval()
        # seed = 1111
        # Note: accuracy may slightly decrease, depending on seed
        # torch.manual_seed(seed)
        preprocessing = dict(mean=self.normalize['mean'], std=self.normalize['std'], axis=-3)
        fmodel = PyTorchModel(self.net, bounds=(0, 1), preprocessing=preprocessing)
        model2 = ModelWrapper(self.net, mean=self.normalize['mean'], std=self.normalize['std'])

        if args.att == 'fgsm':
            attack = fa.FGSM()
        elif args.att == 'pgd':
            attack = fa.LinfPGD()
        elif args.att == 'df':
            attack = fa.L2DeepFoolAttack()
            args.eps = None
        elif args.att == 'cw':
            attack = fa.L2CarliniWagnerAttack(steps=1000)
            args.eps = None
        elif args.att == 'mpgd':
            from openood.attacks import masked_pgd_attack, NormalizeWrapper
            norm_model = NormalizeWrapper(self.net, self.normalize['mean'], self.normalize['std'])

        if self.id_name == 'imagenet':
            attack_path = os.path.join(self.data_root, 'images_largescale', 'imagenet_1k' + '_' + args.att + '_' + args.arch, 'val')
            img_list = "benchmark_imglist/imagenet/test_imagenet.txt"
        elif self.id_name == 'cifar10':
            attack_path = os.path.join(self.data_root, 'images_classic', 'cifar10' + '_' + args.att + '_' + args.arch, 'test')
            img_list = "benchmark_imglist/cifar10/test_cifar10.txt"
        elif self.id_name == 'cifar100':
            attack_path = os.path.join(self.data_root, 'images_classic', 'cifar100' + '_' + args.att + '_' + args.arch, 'test')
            img_list = "benchmark_imglist/cifar100/test_cifar100.txt"
        elif self.id_name == 'imagenet200':
            attack_path = os.path.join(self.data_root, 'images_largescale', 'imagenet200' + '_' + args.att + '_' + args.arch, 'val')
            img_list = "benchmark_imglist/imagenet200/test_imagenet200.txt"
        create_dir(attack_path)

        with open(os.path.join(self.data_root, img_list)) as f:
            lines = f.readlines()

        timestamp_start =  datetime.now().strftime("%Y-%m-%d-%H:%M")

        start_time = time.time()

        counter = 0
        total_samples = 0
        correct_predicted = 0
        successful_attacked = 0

        attacked_target = dict()
        attacked_target["labels"] = []
        attacked_target["correct_predicted"] = 0
        attacked_target["success_full_attacked_target"] = []
        attacked_target["attacked_prediction"] = []
        attacked_target[args.att] = []

        # try:
        for batch in tqdm(self.dataloader_dict['id']['test'], desc="attack", disable=not True):
            data = batch['data'].cuda()
            labels = batch['label'].cuda()
            preds = self.predict(fmodel, data, labels)
            total_samples += len(labels)
            correct_predicted += (labels==preds).cpu().sum().item()

            attacked_target["labels"].append(labels.cpu())
            # attacked_target["correct_predicted"].append(correct_predicted)

            if args.att in DEEPFOOL:
                raw_advs, clipped_advs, success = attack(fmodel, data, criterion=Misclassification(labels), epsilons=args.eps)
            
            if args.att == 'mpgd':
                clipped_advs, success = masked_pgd_attack(norm_model, data, labels, epsilon=1, alpha=0.01, num_steps=40, patch_size=args.masked_patch_size)
            
            if args.att == 'eot_pgd':
                clipped_advs = attack(data, labels)
                pred = torch.max(fmodel(clipped_advs),dim=1)[1]
                success = ~(pred == labels)

            if args.att in ['bandits', 'nes']:
                from blackbox_attacks.bandits import make_adversarial_examples as bandits_attack
                out = bandits_attack(data, labels, args, fmodel, 256)
                success = out['success_adv']
                clipped_advs = out['images_adv']
            
            success = success.cpu()
            attacked_target["success_full_attacked_target"].append(success.cpu())

            att_pred = self.predict(fmodel, clipped_advs, labels)
            attacked_target["attacked_prediction"].append(att_pred.cpu())
            successful_attacked += success.sum().item()
            attacked_target[args.att].append(clipped_advs.cpu())

            for it, suc in enumerate(success):
                clipped_adv = clipped_advs[it].cpu()
        
                image_pil_adv = trn.ToPILImage()(clipped_adv)
                if self.id_name in ["cifar10", "cifar100"]:
                    parse = "/".join(lines[counter].split("/")[2:]).split(" ")[0]
                    create_dir(os.path.join(attack_path, parse.split("/")[0]))
                    image_pil_adv.save(os.path.join(attack_path, f'{parse}'))
                else:
                    parse = lines[counter].split("/")[-1].split(".")[0]
                    image_pil_adv.save(os.path.join(attack_path, f'{parse}.png'))
                counter += 1

            if args.debug: print("Arch: ", args.arch)
            targets = None # [ClassifierOutputTarget(None)]
            if args.arch in ['ResNet18_32x32', 'ResNet18_224x224', 'resnet50']:
                target_layers = [model2.layer4[-1]]
            elif args.arch == 'swin-t':
                # https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/swinT_example.py
                target_layers = [model2.features[-1][-1].norm1]
            elif args.arch == 'vit-b-16':
                # https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/vit_example.py
                target_layers = [model2.encoder.layers[-1]]
            else:
                raise NotImplementedError("Arch not found {}".format(args.arch))


            # Append predictions to the list
            predictions_gt.append(labels.cpu())
            predictions_ben.append(preds.cpu())
            predictions_pgd.append(att_pred.cpu())

        base_pth = os.path.join('./data/attacked', args.att + "_" + self.id_name + "_" + args.arch)
        # save logs
        create_dir(base_pth)
        log_pth = os.path.join(base_pth, 'logs')
        log = create_log_file(args, log_pth)
        log['timestamp_start'] = timestamp_start

        log['successful_attacked'] = successful_attacked
        asr = successful_attacked / total_samples 

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time
        # Convert elapsed time to hours and minutes
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)

        log['load_json'] = "/home"
        log['elapsed_time'] = str(hours) + "h:" + str(minutes) + "m"
        log['total_samples'] = total_samples
        log['correct_predicted'] = correct_predicted
        log['successful_attacked'] = successful_attacked
        log['model_accuracy'] = round(correct_predicted/total_samples, 4)
        log['asr'] = round(asr,4)

        save_log(args, log, log_pth)

        predictions["gt"]  = torch.cat(predictions_gt)
        predictions["ben"] = torch.cat(predictions_ben)
        predictions[args.att] = torch.cat(predictions_pgd)

        # save dictionary
        attacked_target["correct_predicted"] = correct_predicted
        attacked_target["labels"] = torch.cat(attacked_target["labels"])
        attacked_target["success_full_attacked_target"] = torch.cat(attacked_target["success_full_attacked_target"])
        attacked_target["attacked_prediction"] = torch.cat(attacked_target["attacked_prediction"])
        attacked_target[args.att] = torch.vstack(attacked_target[args.att])
        torch.save(attacked_target, os.path.join(base_pth, args.att + "_attacked_target.pth"), pickle_protocol=5)
