import linecache
import json
import os
import random

import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class DrumsAccompanimentDataset(Dataset):

    def __init__(self, file_path, max_seq=2048, vocab_size=318, num_classes=512, batch_size = 1, data_size=1000) -> None:
        self.data = []
        self.file_path = file_path
        self.num_classes = num_classes
        self.max_seq = max_seq
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.data_size = data_size

    def __len__(self):

        lines = 0
        with open(self.file_path, 'r') as f:
            for _ in f:
                lines += 1

        return lines
    
    def __getitem__(self, index):
        # index = 0 causes error linecache.getline(self.file_path, 0) returns empty string
        index = index if index !=0 else 1
        # indices = torch.randperm(self.data_size)[:self.batch_size]
        
        # batch = torch.Tensor(())
        # for index in indices:
        item_dict = json.loads(linecache.getline(self.file_path, int(index)))
        drum_events = torch.LongTensor(item_dict['drums'])
        accomp_events = torch.LongTensor(item_dict['accomp'])

        #drum_tensor = one_hot(drum_events, num_classes=self.num_classes)
        # accomp_tensor = one_hot(accomp_events, num_classes=self.num_classes)
        accomp_tensor, drum_tensor = self.select_seq(accomp_events, drum_events, self.max_seq, self.vocab_size)

        # batch.append((accomp_tensor, drum_tensor))
        return accomp_tensor, drum_tensor

        # return batch
    
    def select_seq(self, accomp_events, drum_events, seq_len, vocab_size):
        TOKEN_START = vocab_size+1
        TOKEN_PAD = vocab_size+2
        TOKEN_END = vocab_size+3
        SEQUENCE_START = 0
        accomp_tensor = torch.full((seq_len,), TOKEN_PAD)
        drum_tensor = torch.full((seq_len,), TOKEN_PAD)
        accomp_tensor[0] = TOKEN_START
        drum_tensor[0] = TOKEN_START

        if len(accomp_events) < seq_len:
            accomp_tensor[len(accomp_events)] = TOKEN_END
            accomp_tensor[:len(accomp_events)] = accomp_events
        else:
            accomp_tensor[seq_len-1] = TOKEN_END
            accomp_tensor[1:seq_len-1] = accomp_events[1:seq_len-1]

        if len(drum_events) < seq_len:
            drum_tensor[len(drum_events)] = TOKEN_END
            drum_tensor[:len(drum_events)] = drum_events
        else:
            drum_tensor[seq_len-1] = TOKEN_END
            drum_tensor[1:seq_len-1] = drum_events[1:seq_len-1]

        return accomp_tensor, drum_tensor
    
    
def create_accomp_drum_dataset(dataset_path, max_seq=2048, random_seq=False, vocab_size=None):

    dataset = DrumsAccompanimentDataset(dataset_path, max_seq=max_seq, vocab_size=vocab_size)
    print('length of dataset: ',len(dataset))

    # train_subset, val_subset, test_subset = torch.utils.data.random_split(dataset, [800, 100, 100])
    train_dataset = dataset

    # TODO: Get train, val, test splits
    return train_dataset
