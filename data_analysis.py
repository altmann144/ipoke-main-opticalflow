import torch
import yaml
from data.datamodule import StaticDataModule

import numpy as np

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    with open('config/BigGanAE.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    #
    datakeys = ['flow']
    config['data']['batch_size'] = 40
    config["architecture"]["in_size"] = config["data"]["spatial_size"][0]
    datamod = StaticDataModule(config['data'], datakeys=datakeys)
    datamod.setup()

    for dset_type, dataloader in {'val': datamod.val_dataloader(), 'train': datamod.train_dataloader()}.items():
        max_val = 0
        min_val = 0
        median = 0
        mean = 0
        N = len(dataloader)
        for batch in dataloader:
            flow = batch['flow']
            min_val = min(torch.min(flow).item(), min_val)
            max_val = max(torch.max(flow).item(), max_val)
            mean += torch.mean(flow)/N
            median += 1e-3 * np.sign(torch.median(flow).item() - median)

        print(dset_type)
        print(f'{"mean:":<8}: {mean}')
        print(f'{"median:":<8}: {median}')
        print(f'{"min:":<8} {min_val}')
        print(f'{"max:":<8}: {max_val}')
        print()

    # with open('scratch/FCAE_eval.json', 'w+') as f:
    #     json.dump(R, f, indent=4)
    # wandb_logger.experiment.log(R)

