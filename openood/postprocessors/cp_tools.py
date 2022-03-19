from openood.datasets import get_dataloader
from openood.utils import setup_logger
import torch
from tqdm import tqdm

def get_train_embeds(net, config):
    # setup_logger(config)
        
    preprocessor = None
    loader_dict = get_dataloader(config.dataset, preprocessor)
    train_loader = loader_dict['train']
    # test_loader = loader_dict['test']

    train_embed = []
    train_dataiter = iter(train_loader)
    with torch.no_grad():
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                                     desc='Train embeds:'):
            batch = next(train_dataiter)
            # data = torch.cat(batch['data'], 0)
            # data = data.cuda()
            data = batch['data'].cuda()
            
            embed, logit = net(data)

            train_embed.append(embed.cuda())

    train_embed = torch.cat(train_embed)

    return train_embed