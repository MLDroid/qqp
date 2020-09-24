import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm

import config


class dataset(Dataset):
    def __init__(self, sentence_pairs, labels, max_len):
        self.model_name = config.MODEL_NAME
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        print(f'Set max seq. len: {self.max_len} for tokenizer: {self.tokenizer}')

        self.sent_token_ids_attn_masks = [self._get_token_ids_attn_mask(s_pair, lower=config.IS_LOWER)
                                          for s_pair in tqdm(sentence_pairs)]
        self.labels = np.array(labels,dtype=int)
        print(f'Loaded X_train and y_train, shapes: {len(self.sent_token_ids_attn_masks), self.labels.shape}')


    def _get_token_ids_attn_mask(self, s_pair, lower=False):
        s1 = str(s_pair[0]).strip()
        s2 = str(s_pair[1]).strip()
        s1 = ' '.join(s1.split())  # make sure unwanted spaces are removed
        s2 = ' '.join(s2.split())  # make sure unwanted spaces are removed
        if lower:
            s1 = s1.lower()
            s2 = s2.lower()

        # encode_plus is better than calling tokenizer.tokenize and get the IDs later -
        # ref:Abisek Thakur youtube video
        inputs = self.tokenizer.encode_plus(s1, s2,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            padding='max_length',
                                            truncation=True
                                            )

        #need to convert them as tensors
        tokens_ids_tensor = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attn_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        return tokens_ids_tensor, attn_mask


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        #Selecting the sentence and label at the specified index in the data frame
        token_ids,attn_mask = self.sent_token_ids_attn_masks[index] #list index
        label = self.labels[index] #array index
        return token_ids, attn_mask, label



class test_dataset(Dataset):
    def __init__(self, sentence_pairs, max_len):
        self.model_name = config.MODEL_NAME
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        print(f'Set max seq. len: {self.max_len} for tokenizer: {self.tokenizer}')

        self.sent_token_ids_attn_masks = [self._get_token_ids_attn_mask(s_pair, lower=config.IS_LOWER)
                                          for s_pair in tqdm(sentence_pairs)]
        print(f'Loaded X_test, shape: {len(self.sent_token_ids_attn_masks)}')


    def _get_token_ids_attn_mask(self, s_pair, lower=False):
        s1 = str(s_pair[0]).strip()
        s2 = str(s_pair[1]).strip()
        s1 = ' '.join(s1.split())  # make sure unwanted spaces are removed
        s2 = ' '.join(s2.split())  # make sure unwanted spaces are removed
        if lower:
            s1 = s1.lower()
            s2 = s2.lower()

        # encode_plus is better than calling tokenizer.tokenize and get the IDs later -
        # ref:Abisek Thakur youtube video
        inputs = self.tokenizer.encode_plus(s1, s2,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            pad_to_max_length=True,
                                            truncation=True
                                            )

        #need to convert them as tensors
        tokens_ids_tensor = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attn_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        return tokens_ids_tensor, attn_mask



    def __getitem__(self, index):
        #Selecting the sentence and label at the specified index in the data frame
        token_ids,attn_mask = self.sent_token_ids_attn_masks[index] #list index
        return token_ids, attn_mask
