import csv
from itertools import islice
import os
from typing import Dict, Iterable, Iterator, List, Literal, TypeVar

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from openood.evaluators.ood_evaluator import OODEvaluator
from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .metrics import (
    auc_and_fpr_recall,
    acc,
    compute_der,
    any_float,
    detection_error_rate,
)

U = TypeVar("U")


def batched(iterable: Iterable[U], n: int) -> Iterator[tuple[U, ...]]:
    # from python 3.12
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def compute_openood_metrics(conf: np.ndarray, label: np.ndarray, pred: np.ndarray) -> Dict[str, any_float]:
    np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)

    accuracy = acc(pred, label)

    return {
        "FPR@95": fpr,
        "AUROC": auroc,
        "AUPR_IN": aupr_in,
        "AUPR_OUT": aupr_out,
        "ACC": accuracy,
    }


class HCOODEvaluator(OODEvaluator):
    def __init__(self, config: Config):
        """OOD Evaluator.

        Args:
            config (Config): Config file from
        """
        super(HCOODEvaluator, self).__init__(config)
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None

    def eval_ood(
        self,
        net: nn.Module,
        id_data_loaders: Dict[str, DataLoader],
        ood_data_loaders: Dict[str, Dict[str, DataLoader]],
        postprocessor: BasePostprocessor,
        fsood: bool = False,
    ):
        if isinstance(net, dict):
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert "test" in id_data_loaders, "id_data_loaders should have the key: test!"
        dataset_name = self.config.dataset.name

        if self.config.postprocessor.APS_mode:
            assert "val" in id_data_loaders
            assert "val" in ood_data_loaders
            self.hyperparam_search(
                net, id_data_loaders["val"], ood_data_loaders["val"], postprocessor
            )

        print(f"Performing inference on {dataset_name} dataset...", flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(net, id_data_loaders["test"])
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        if fsood:
            # load csid data and compute confidence
            for dataset_name, csid_dl in ood_data_loaders["csid"].items():
                print(f"Performing inference on {dataset_name} dataset...", flush=True)
                csid_pred, csid_conf, csid_gt = postprocessor.inference(net, csid_dl)
                if self.config.recorder.save_scores:
                    self._save_scores(csid_pred, csid_conf, csid_gt, dataset_name)
                id_pred = np.concatenate([id_pred, csid_pred])
                id_conf = np.concatenate([id_conf, csid_conf])
                id_gt = np.concatenate([id_gt, csid_gt])

        # load nearood data and compute ood metrics
        self.print_separator()
        self._eval_ood(
            net,
            [id_pred, id_conf, id_gt],
            ood_data_loaders,
            postprocessor,
            ood_split="nearood",
        )

        # load farood data and compute ood metrics
        self.print_separator()
        self._eval_ood(
            net,
            [id_pred, id_conf, id_gt],
            ood_data_loaders,
            postprocessor,
            ood_split="farood",
        )

    def _eval_ood(
        self,
        net: nn.Module,
        id_list: List[np.ndarray],
        ood_data_loaders: Dict[str, Dict[str, DataLoader]],
        postprocessor: BasePostprocessor,
        ood_split: str = "nearood",
    ):
        print(f"Processing {ood_split}...", flush=True)
        [id_pred, id_conf, id_gt] = id_list
        gammas = {
            p: np.quantile(id_conf, q=p) for p in [0.95, 0.99]
        }  # DER95, DER99 gammas
        all_pred = [id_pred]
        all_conf = [id_conf]
        all_gt = [id_gt]
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            print(f"Performing inference on {dataset_name} dataset...", flush=True)
            ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_dl)
            all_pred.append(ood_pred)
            all_conf.append(ood_conf)
            all_gt.append(ood_gt)

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            if self.config.recorder.save_scores:
                self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])


            print(f"Computing metrics on {dataset_name} dataset...")

            ood_metrics = compute_openood_metrics(conf, label, pred)
            for gamma_name, gamma in gammas.items():
                ood_metrics[f"DER{100*gamma_name:.0f}"] = detection_error_rate(
                    y_cor=(ood_gt == ood_pred), ood_conf=ood_conf, gamma=gamma
                )

            if self.config.recorder.save_csv:
                self._print_metrics(ood_metrics)
                self.print_separator()
                self._save_csv(ood_metrics, dataset_name=dataset_name)

        print("Computing aggregated metrics...", flush=True)
        all_pred = np.concatenate(all_pred)
        all_conf = np.concatenate(all_conf)
        all_gt = np.concatenate(all_gt)
        ood_metrics = compute_openood_metrics(all_conf, all_gt, all_pred)
        for gamma_name, gamma in gammas.items():
            ood_metrics[f"DER{100*gamma_name:.0f}"] = detection_error_rate(
                y_cor=(all_gt == all_pred), ood_conf=all_conf, gamma=gamma
            )
        self._print_metrics(ood_metrics)
        if self.config.recorder.save_csv:
            self._save_csv(ood_metrics, dataset_name=ood_split)

    def eval_ood_val(
        self,
        net: nn.Module,
        id_data_loaders: Dict[str, DataLoader],
        ood_data_loaders: Dict[str, DataLoader],
        postprocessor: BasePostprocessor,
    ) -> Dict[Literal["auroc"], any_float]:
        if isinstance(net, dict):
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert "val" in id_data_loaders
        assert "val" in ood_data_loaders
        if self.config.postprocessor.APS_mode:
            val_auroc = self.hyperparam_search(
                net, id_data_loaders["val"], ood_data_loaders["val"], postprocessor
            )
        else:
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loaders["val"]
            )
            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, ood_data_loaders["val"]
            )
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_openood_metrics(conf, label, pred)
            val_auroc = ood_metrics["AUROC"]
        return {"auroc": 100 * val_auroc}

    def _save_csv(self, metrics: dict[str, any_float], dataset_name: str):
        write_content = {"dataset": dataset_name} | metrics
        fieldnames = list(write_content.keys())
        csv_path = os.path.join(self.config.output_dir, "ood.csv")
        if not os.path.exists(csv_path):
            # create csv file and write header
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            # append to csv file
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _print_metrics(self, metrics: dict[str, any_float]):
        for report_line in batched(metrics.items(), 2):
            print(
                ", ".join(f"{name}: {100 * value:.2f}" for name, value in report_line),
                flush=True,
            )

    @staticmethod
    def print_separator():
        print("\u2500" * 70, flush=True)

    def _save_scores(self, pred, conf, gt, save_name):
        save_dir = os.path.join(self.config.output_dir, "scores")
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name), pred=pred, conf=conf, label=gt)

    def hyperparam_search(
        self,
        net: nn.Module,
        id_data_loader,
        ood_data_loader,
        postprocessor: BasePostprocessor,
    ):
        print("Starting automatic parameter search...")
        aps_dict = {}
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0
        for name in postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1
        for name in hyperparam_names:
            hyperparam_list.append(postprocessor.args_dict[name])
        hyperparam_combination = self.recursive_generator(hyperparam_list, count)
        for hyperparam in hyperparam_combination:
            postprocessor.set_hyperparam(hyperparam)
            id_pred, id_conf, id_gt = postprocessor.inference(net, id_data_loader)
            ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_data_loader)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_openood_metrics(conf, label, pred)
            index = hyperparam_combination.index(hyperparam)
            aps_dict[index] = ood_metrics["AUROC"]
            print("Hyperparam:{}, auroc:{}".format(hyperparam, aps_dict[index]))
            if ood_metrics["AUROC"] > max_auroc:
                max_auroc = ood_metrics["AUROC"]
        for key in aps_dict.keys():
            if aps_dict[key] == max_auroc:
                postprocessor.set_hyperparam(hyperparam_combination[key])
        print("Final hyperparam: {}".format(postprocessor.get_hyperparam()))
        return max_auroc

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results
