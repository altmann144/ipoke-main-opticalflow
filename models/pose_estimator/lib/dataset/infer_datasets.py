import os
from os import path
import pickle
import numpy as np
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import cv2


class InferenceDataset(Dataset):

    def __init__(self,dataset):
        super().__init__()
        self.dataset = dataset
        self.transforms = T.Compose([T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        assert self.dataset in ['IperDataset','TaichiDataset']

        if self.dataset == 'IperDataset':
            self.metafile = 'data/IperDataset/meta.p'
            self.datapath='data/IperDataset'
        else:
            self.metafile = 'data/TaichiDataset/meta.p'
            self.datapath = 'data/TaichiDataset'

        with open(self.metafile,'rb') as f:
            self.data = pickle.load(f)


        self.data = {key: np.asarray(self.data[key]) for key in self.data}

        self.data.update({'did':np.arange(self.data['img_path'].shape[0])})




    def __len__(self):
        return self.data['img_path'].shape[0]

    def __getitem__(self, id):

        return {'id' : self._get_id(id),
                'img' : self._get_img(id)}

    def _get_id(self,idx):
        return self.data['did'][idx]

    def _get_img(self,idx):

        img_path = self.data['img_path'][idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,dsize=(256,256),interpolation=cv2.INTER_LINEAR)


        img = self.transforms(img)
        return img
