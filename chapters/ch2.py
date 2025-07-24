from importlib.metadata import version
import torch
import tiktoken
import os
import urllib.request
from torch.utils.data import DataLoader, Dataset
import numpy as np
#print("torch version:", version("torch"))
#print("tiktoken version:", version("tiktoken"))


text_file = "../hltv-forum-scraper/hltv_forum_threads.txt"

with open(text_file, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

encoded_text = tokenizer.encode(text)
print(len(encoded_text))

context_size = 4

x = encoded_text[:context_size]
y = encoded_text[1:context_size+1]

#print(f"x: {x}")
#print(f"y: {y}")

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        #sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    Create a dataloader for the GPT dataset.
    Args: text (str): The text to create the dataset from
          batch_size (int): The batch size (set equal to the context size)
          max_length (int): The maximum length of the input sequence
          stride (int): The number of positions to shift the window
          shuffle (bool): Whether to shuffle the dataset
          drop_last (bool): Whether to drop the last batch if it's not full (if dataset size is not divisible by batch size, last batch will be smaller, so set to True to prevent loss
          num_workers (int): The number of workers to use for the dataloader
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


dataloader = create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=False, drop_last=True, num_workers=0)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
#print(tokenizer.n_vocab)

embedding_layer = torch.nn.Embedding(tokenizer.n_vocab, 256)

token_embeddings = embedding_layer(inputs)
print(token_embeddings.shape)
#max length from above is used as the embedding dimension
max_length = 256
positional_embedding_layer = torch.nn.Embedding(max_length, 256)

print(positional_embedding_layer.weight)

positional_embeddings = positional_embedding_layer(torch.arange(max_length))
print(positional_embeddings.shape)

input_embeddings = token_embeddings + positional_embeddings
print(input_embeddings.shape)
