import logging
import pickle
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd

class MMDataset(Dataset):
    def __init__(self, args):
        self.args = args

        DATASET_MAP = {
            'Expertise': self.__init_evaluation}
        DATASET_MAP[args['dataset_name']]()

    def __init_evaluation(self):

        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)


        self.text = data['text']
        self.vision = data['vision']
        self.audio = data['audio']
        try:
           self.info = data['info'].tolist()
        except:
            self.info = data['info']
        self.raw_text = data['raw_text'].tolist()
        self.labels = {'M': np.array(data['labels']-4).astype(np.float32)}

        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0
        self.text[self.text != self.text] = 0


        self.args['feature_dims'][0] = self.text[1].shape[-1]
        self.args['feature_dims'][1] = self.audio[1].shape[-1]
        self.args['feature_dims'][2] = self.vision[1].shape[-1]


        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # For visual and audio modality, we average across time:
        # The original data has shape (max_len, num_examples, feature_dim)
        # After averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)
        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return  self.text.shape[0]

    def get_seq_len(self):
        return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
        'text': torch.Tensor(self.text[index]),
        'audio': torch.Tensor(self.audio[index]),
        'vision': torch.Tensor(self.vision[index]),
        'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
        'raw_text': self.raw_text[index],
        "meta_info": self.info[index]
        }
        return sample


def MMDataLoader(args, num_workers=0):
    # Create datasets
    datasets =   MMDataset(args)
    # Get the sequence lengths for each modality if specified in the arguments
    args['seq_lens'] = datasets.get_seq_len()
    # Create DataLoader objects for each split
    dataLoader =  DataLoader(datasets,
                       batch_size=128,
                       num_workers=num_workers,
                       shuffle=False)
    return dataLoader
