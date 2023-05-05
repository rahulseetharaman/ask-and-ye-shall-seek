from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv
import pandas as pd

class ClariqDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.all_texts = pd.read_csv(data_path, sep='\t', header=None)
        self.contexts = self.all_texts.iloc[:, 0].values.tolist()
        self.clques = self.all_texts.iloc[:, 1].values.tolist()
        encodings = tokenizer.prepare_seq2seq_batch(self.contexts, self.clques, max_length=512, padding=True, truncation=True, return_tensors="pt")
        self.encoder_input_ids = encodings.input_ids
        self.encoder_attention_mask = encodings.attention_mask
        self.decoder_input_ids = encodings.labels[:, :-1].clone()  # skip last
        self.labels = encodings.labels[:, 1:].clone()              # skip first

    def __len__(self):
        return len(self.all_texts)

    def __getitem__(self, item):
        return self.encoder_input_ids[item], self.encoder_attention_mask[item], self.decoder_input_ids[item], self.labels[item] 