from openood.datasets import get_dataloader
from openood.utils import setup_logger


def get_train_embeds(net, config):
    setup_logger(config)
    
    preprocessor = None
    loader_dict = get_dataloader(config.dataset, preprocessor)
    train_loader, val_loader = loader_dict['train'], loader_dict['val']
    # test_loader = loader_dict['test']
    
    train_embed = []
    # train_dataiter = iter(train_loader)
    with torch.no_grad():
        for x in train_loader:
            embed, logit = net(x.to(device))

            train_embed.append(embed.cpu())
    train_embed = torch.cat(train_embed)
    return train_embed
