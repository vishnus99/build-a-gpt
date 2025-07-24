import torch
import torch.nn as nn
import tiktoken
import os
import urllib.request
from torch.utils.data import DataLoader, Dataset
import numpy as np

#Implementing a GPT model from scratch to generate text


#Config for the 124 million parameter GPT-2 model

GPT2_CONFIG ={
    "vocab_size": 50257, #vocab size, supported by BPE tokenizer
    "context_length": 1024, #maximum input token count
    "embedding_dim": 768, #embedding size for token inputs, each token is converted to a 768 dimensional vector
    "n_heads": 12, #number of attention heads in multi-head attention mechanism
    "n_layers": 12, #number of transformer layers in the model
    "dropout_rate": 0.1, #dropout rate, 10% of hidden units are dropped to mitigate overfitting
    "qkv_bias": False #decides if Linear layers in the multi-head attention mechanism should include a bias vector when computing Q, K, V tensors. Disabling this is standard practice for LLMs

}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.drop_emb = nn.Dropout(cfg["dropout_rate"])
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["embedding_dim"])
        self.out_head = nn.Linear(
            cfg["embedding_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x
    

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)
print ("Batch: ", batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT2_CONFIG)
logits = model(batch)

print("Output shape: ", logits.shape)
