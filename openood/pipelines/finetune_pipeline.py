import argparse
import shutil
import time
from functools import partial
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from fsood.data import get_dataloader
from fsood.evaluation import Evaluator
from fsood.networks import get_network
from fsood.trainers import get_trainer
from fsood.utils import load_yaml, setup_logger


def main(args, config):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # expt_output_dir = (output_dir / config["name"]).mkdir(parents=True, exist_ok=True)

    # Save a copy of config file in output directory
    config_path = Path(args.config)
    config_save_path = output_dir / 'config.yml'
    shutil.copy(config_path, config_save_path)

    setup_logger(str(output_dir))

    benchmark = config['dataset']['benchmark']
    if benchmark == 'DIGITS':
        num_classes = 10
    elif benchmark == 'OBJECTS':
        num_classes = 10
    elif benchmark == 'COVID':
        num_classes = 2
    else:
        raise ValueError('Unknown Benchmark!')

    # Init Datasets ############################################################
    get_dataloader_default = partial(
        get_dataloader,
        root_dir=args.data_dir,
        benchmark=benchmark,
        num_classes=num_classes,
        image_size=config['dataset']['image_size'],
    )

    train_loader = get_dataloader_default(
        name=config['dataset']['train_id'],
        stage='train',
        batch_size=config['dataset']['labeled_batch_size'],
        shuffle=True,
        num_workers=args.prefetch,
    )

    test_id_loader = get_dataloader_default(
        name=config['dataset']['train_id'],
        stage='test',
        batch_size=config['dataset']['test_batch_size'],
        shuffle=False,
        num_workers=args.prefetch,
    )

    test_ood_loader_list = []
    for name in config['dataset']['test_ood']:
        test_ood_loader = get_dataloader_default(
            name=name,
            stage='test',
            batch_size=config['dataset']['test_batch_size'],
            shuffle=False,
            num_workers=args.prefetch,
        )
        test_ood_loader_list.append(test_ood_loader)

    # Init Network #############################################################
    net = get_network(
        config['network'],
        num_classes,
        checkpoint=args.checkpoint,
    )

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True  # fire on all cylinders

    # Init Trainer #############################################################
    trainer = get_trainer(
        config['trainer_name'],
        net,
        train_loader,
        config['optim_args'],
        config['trainer_args'],
    )

    # Start Training ###########################################################
    evaluator = Evaluator(net)

    output_dir = Path(args.output_dir)

    begin_epoch = time.time()
    best_accuracy = 0.0
    for epoch in range(0, config['optim_args']['epochs']):

        train_metrics = trainer.train_epoch()

        classification_metrics = evaluator.eval_classification(test_id_loader)
        # evaluator.eval_ood(
        #     test_id_loader,
        #     test_ood_loader_list,
        #     method="full",
        # )

        # Save model
        torch.save(net.state_dict(), output_dir / f'epoch_{epoch}.ckpt')
        if not args.save_all_model:
            # Let us not waste space and delete the previous model
            prev_path = output_dir / f'epoch_{epoch - 1}.ckpt'
            prev_path.unlink(missing_ok=True)

        # save best result
        if classification_metrics['test_accuracy'] >= best_accuracy:
            torch.save(net.state_dict(), output_dir / f'best.ckpt')

            best_accuracy = classification_metrics['test_accuracy']

        print(
            'Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.3f} | Test Acc {:.2f}'
            .format(
                (epoch + 1),
                int(time.time() - begin_epoch),
                train_metrics['train_loss'],
                classification_metrics['test_loss'],
                100.0 * classification_metrics['test_accuracy'],
            ),
            flush=True,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        help='path to config file',
        default='configs/train/cifar10_udg.yml',
    )
    parser.add_argument(
        '--checkpoint',
        help='specify path to checkpoint if loading from pre-trained model',
    )
    parser.add_argument(
        '--data_dir',
        help='directory to dataset',
        default='data',
    )
    parser.add_argument(
        '--output_dir',
        help='directory to save experiment artifacts',
        default='output/cifar10_udg',
    )
    parser.add_argument(
        '--save_all_model',
        action='store_true',
        help='whether to save all model checkpoints',
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
