import torch
from torch.utils.data import Dataset
import pandas as pd


class WordsDataSet(Dataset):

    def __init__(self, df):
        self.words = df['word'].tolist()
        self.labels = df['tag'].tolist()
        self.words_vec = df['new_vector'].to_list()
        self.tags_to_idx = {tag: idx for idx, tag in enumerate(sorted(list(set(self.labels))))}
        self.idx_to_tag = {idx: tag for tag, idx in self.tags_to_idx.items()}
        self.vocabulary_size = len(df['new_vector'][0])

    def __getitem__(self, item):
        cur_vec = self.words_vec[item]
        cur_vec = torch.FloatTensor(cur_vec).squeeze()
        label = self.labels[item]
        label = self.tags_to_idx[label]
        # label = torch.Tensor(label)
        data = {"input_ids": cur_vec, "labels": label}
        return data

    def __len__(self):
        return len(self.words)
