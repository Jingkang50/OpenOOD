import argparse
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from fsood.data import get_dataloader
from fsood.evaluation import Evaluator
from fsood.networks import get_network
from fsood.postprocessors import get_postprocessor
from fsood.utils import load_yaml
from torch.functional import Tensor
from tqdm import tqdm


def main(args, config):
    benchmark = config['benchmark']
    if benchmark == 'DIGITS':
        num_classes = 10
    elif benchmark == 'OBJECTS':
        num_classes = 10
    elif benchmark == 'COVID':
        num_classes = 2
    else:
        raise ValueError('Unknown Benchmark!')

    # Init Datasets ###########################################################
    print('Initializing Datasets...')
    get_dataloader_default = partial(
        get_dataloader,
        root_dir=args.data_dir,
        benchmark=benchmark,
        num_classes=num_classes,
        stage='test',
        interpolation=config['interpolation'],
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=args.prefetch,
    )

    val_id_loader = get_dataloader_default(name=config['id_dataset'],
                                           stage='val')
    val_ood_loader = get_dataloader_default(name=config['val_dataset'],
                                            stage='val')

    test_id_loader = get_dataloader_default(name=config['id_dataset'])

    test_csid_loader_list = []
    for name in config['csid_datasets']:
        test_ood_loader = get_dataloader_default(name=name)
        test_csid_loader_list.append(test_ood_loader)

    test_nearood_loader_list = []
    for name in config['nearood_datasets']:
        test_ood_loader = get_dataloader_default(name=name)
        test_nearood_loader_list.append(test_ood_loader)

    test_farood_loader_list = []
    for name in config['farood_datasets']:
        test_ood_loader = get_dataloader_default(name=name)
        test_farood_loader_list.append(test_ood_loader)

    # Init Network ############################################################
    print('Initializing Network...')
    net = get_network(
        config['network'],
        num_classes,
        checkpoint=args.checkpoint,
    )
    net.eval()

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True

    # Init Evaluator
    print('Starting Evaluation...')
    # Init postprocessor
    postprocess_args = config['postprocess_args'] if config[
        'postprocess_args'] else {}

    if config['postprocess'] == 'gmm':
        # step 1: estimate initial mean and variance from training set
        from fsood.postprocessors import get_GMM_stat, alpha_selector
        train_loader = get_dataloader_default(name=config['id_dataset'],
                                              stage='val')
        gmm_estimator_args = {
            'num_clusters_list': postprocess_args['num_clusters_list'],
            'feature_type_list': postprocess_args['feature_type_list'],
            'feature_process_list': postprocess_args['feature_process_list']
        }
        feature_mean, feature_prec, component_weight, trans_mat = get_GMM_stat(
            train_loader, net, **gmm_estimator_args)

        # step 2: hyperparam searching on alpha
        if postprocess_args['alpha_list']:
            print('Load predefined alpha list...')
            alpha_list = postprocess_args['alpha_list']
        else:
            print('Searching for optimal alpha list...')
            gmm_alpha_search_args = {
                'feature_type_list': postprocess_args['feature_type_list'],
                'feature_mean_list': feature_mean,
                'feature_prec_list': feature_prec,
                'component_weight_list': component_weight,
                'transform_matrix_list': trans_mat,
                'alpha_list': []
            }
            gmm_alpha_search_postprocessor = get_postprocessor(
                'gmm', **gmm_alpha_search_args)
            scores_in, scores_out = None, None
            for batch in tqdm(val_id_loader,
                              desc=f'val_{val_id_loader.dataset.name}'):
                data = batch['plain_data'].cuda()
                score_in = gmm_alpha_search_postprocessor(net,
                                                          data,
                                                          return_scores=True)
                if scores_in == None:
                    scores_in = score_in
                else:
                    scores_in = torch.cat((scores_in, score_in), dim=0)
            for batch in tqdm(val_ood_loader,
                              desc=f'val_{val_ood_loader.dataset.name}'):
                data = batch['plain_data'].cuda()
                score_out = gmm_alpha_search_postprocessor(net,
                                                           data,
                                                           return_scores=True)
                if scores_out == None:
                    scores_out = score_out
                else:
                    scores_out = torch.cat((scores_out, score_out), dim=0)
            scores_in = scores_in.data.cpu().numpy()
            scores_out = scores_out.data.cpu().numpy()
            # logistic regression for optimal alpha
            alpha_list = alpha_selector(scores_in, scores_out)

        # step 3: postprocessor with hyperparam
        gmm_score_args = {
            'feature_type_list': postprocess_args['feature_type_list'],
            'feature_mean': feature_mean,
            'feature_prec': feature_prec,
            'component_weight': component_weight,
            'transform_matrix': trans_mat,
            'alpha_list': alpha_list
        }
        postprocessor = get_postprocessor('gmm', **gmm_score_args)

    elif config['postprocess'] == 'mds':
        # step 1: estimate initial mean and variance from training set
        from fsood.postprocessors import sample_estimator, \
            get_Mahalanobis_score, alpha_selector
        train_loader = get_dataloader_default(name=config['id_dataset'],
                                              stage='train')
        num_layer = len(postprocess_args['feature_type_list'])
        sample_estimator_args = {
            'model': net,
            'num_classes': num_classes,
            'train_loader': train_loader,
            'feature_type_list': postprocess_args['feature_type_list'],
            'feature_process_list': postprocess_args['feature_process_list']
        }
        feature_mean, feature_prec = sample_estimator(**sample_estimator_args)

        # step 2: input process and logistic regression for hyperparam search
        magnitude = postprocess_args['noise']
        if postprocess_args['alpha_list']:
            print('Load predefined alpha list...')
            alpha_optimal = postprocess_args['alpha_list']
        else:
            print('Searching for optimal alpha list...')
            for layer_index in range(num_layer):
                M_in = get_Mahalanobis_score(net, val_id_loader, num_classes,
                                             feature_mean, feature_prec,
                                             layer_index, magnitude)
                M_in = np.asarray(
                    M_in, dtype=np.float32)  # convert to numpy for sklearn
                if layer_index == 0:
                    Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
                else:
                    Mahalanobis_in = np.concatenate(
                        (Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))),
                        axis=1)
            for layer_index in range(num_layer):
                M_out = get_Mahalanobis_score(net, val_ood_loader, num_classes,
                                              feature_mean, feature_prec,
                                              layer_index, magnitude)
                M_out = np.asarray(
                    M_out, dtype=np.float32)  # convert to numpy for sklearn
                if layer_index == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate(
                        (Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))),
                        axis=1)
            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)

            # logistic regression for optimal alpha
            alpha_optimal = alpha_selector(Mahalanobis_in, Mahalanobis_out)

        # step 3: postprocessor with hyperparam
        postprocess_args = {
            'feature_mean': feature_mean,
            'feature_prec': feature_prec,
            'magnitude': magnitude,
            'alpha_optimal': alpha_optimal
        }
        postprocessor = get_postprocessor(config['postprocess'],
                                          **postprocess_args)

    else:
        postprocessor = get_postprocessor(config['postprocess'],
                                          **postprocess_args)

    evaluator = Evaluator(net)

    print('###################################################')
    print('# In Distribution Evaluation')
    print('###################################################')
    # evaluate ID classification
    evaluator.eval_csid_classification([test_id_loader],
                                       csv_path=args.csv_path)

    # evaluate CSID classification
    evaluator.eval_csid_classification(test_csid_loader_list,
                                       csv_path=args.csv_path)

    # evaluate id covariate shift
    evaluator.eval_ood(
        [test_id_loader],
        test_csid_loader_list,
        postprocessor=postprocessor,
        method=config['eval_method'],
        csv_path=args.csv_path,
    )

    print('###################################################')
    print('# Classic OOD Evaluation')
    print('###################################################')
    # evaluate near OOD
    evaluator.eval_ood(
        [test_id_loader],
        test_nearood_loader_list,
        postprocessor=postprocessor,
        method=config['eval_method'],
        csv_path=args.csv_path,
    )

    # evaluate far OOD
    evaluator.eval_ood(
        [test_id_loader],
        test_farood_loader_list,
        postprocessor=postprocessor,
        method=config['eval_method'],
        csv_path=args.csv_path,
    )

    print('###################################################')
    print('# F-OOD Evaluation')
    print('###################################################')
    # combine all id loaders
    test_id_loader_list = [test_id_loader] + test_csid_loader_list
    # evaluate near OOD
    evaluator.eval_ood(
        test_id_loader_list,
        test_nearood_loader_list,
        postprocessor=postprocessor,
        method=config['eval_method'],
        csv_path=args.csv_path,
    )

    # evaluate far OOD
    evaluator.eval_ood(
        test_id_loader_list,
        test_farood_loader_list,
        postprocessor=postprocessor,
        method=config['eval_method'],
        csv_path=args.csv_path,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        help='path to config file',
        default='configs/test/digits_msp.yml',
    )
    parser.add_argument(
        '--checkpoint',
        help='path to model checkpoint',
        default='output/net.ckpt',
    )
    parser.add_argument(
        '--data_dir',
        help='directory to dataset',
        default='data',
    )
    parser.add_argument(
        '--csv_path',
        help='path to save evaluation results',
        default='results.csv',
    )
    parser.add_argument('--ngpu',
                        type=int,
                        default=1,
                        help='number of GPUs to use')
    parser.add_argument('--prefetch',
                        type=int,
                        default=4,
                        help='pre-fetching threads.')

    args = parser.parse_args()

    # Load config file
    config = load_yaml(args.config)

    main(args, config)
