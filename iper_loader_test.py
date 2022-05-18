import torch
import yaml
from data.datamodule import StaticDataModule
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import flow_vis

if __name__ == '__main__':
    # np.random.seed(42)
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    #
    # device = f"cpu"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ''
    # config_path = 'config/FullyConnected/first_stage.yaml'
    # with open(config_path, 'r') as stream:
    #     config = yaml.load(stream, Loader=yaml.FullLoader)
    #
    # datakeys = ['flow']
    # config['data']['batch_size'] = 1
    # datamod = StaticDataModule(config['data'], datakeys=datakeys, debug=True)
    #
    # # Logger
    # datamod.setup()
    #
    # loader = datamod.train_dataloader()
    # for batch in loader:
    #     flow = batch['flow']
    #     break
    # # with open('/export/scratch/compvis/datasets/plants/processed_256_resized/plants_dataset_daniel.p', 'rb') as infile:
    # #     data_frame = pickle.load(infile)
    files = ['prediction_115_125.flow.npy',
             'prediction_116_126.flow.npy',
             'prediction_117_127.flow.npy',
             'prediction_118_128.flow.npy',
             'prediction_119_129.flow.npy',
             'prediction_120_130.flow.npy',
             'prediction_121_131.flow.npy',
             'prediction_122_132.flow.npy']

    flows = [np.load("/export/scratch/compvis/datasets/iPER/processed_256_resized/001_12_2/"+file) for file in files]

    fig, axes = plt.subplots(1)
    for i in range(1):
        x = flows[3+i]
        x[np.isnan(x)] = 0
        x = x.transpose([1, 2, 0])
        x = flow_vis.flow_to_color(x)
        axes.imshow(x)
    plt.savefig('flow_vis_test.pdf', format='pdf')