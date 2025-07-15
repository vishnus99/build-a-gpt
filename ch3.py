from importlib.metadata import version
import torch
import tiktoken
import os
import urllib.request
from torch.utils.data import DataLoader, Dataset
import numpy as np

#Coding attention mechanisms

#Simplified self-attention -> self-attention -> causal attention -> multi-head attention

"""
The goal of self-attention is to compute a context vector for each input element that combines information from all other input elements.
The importance or contribution of each input element for computing the context vector is determined by the attention weights of the input elements.
When computing the context vector, the attention weights are calculated with respect to the input elements. 
"""


#Simple self attention mechanism without trainable weights

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], #x^1
    [0.55, 0.87, 0.66], #x^2
    [0.78, 0.23, 0.99], #x^3
    [0.91, 0.45, 0.32], #x^4
    [0.21, 0.67, 0.55], #x^5
    [0.33, 0.79, 0.12]] #x^6
)

input_query = inputs[1]
input_1 = inputs[0]

#print(torch.dot(input_query, input_1))


##A simple self attention mechanism without trainable weights

attn_scores = torch.empty(6,6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

#print(attn_scores)

attn_scores = inputs @ inputs.T
#print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=1)
#print(attn_weights)

## Computing the attention weights
x_2 = inputs[1]
print(x_2)
d_in = inputs.shape[1]
print(d_in)
d_out = 2
torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
print(W_query)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

query_2 = x_2 @ W_query

print(query_2)

keys = inputs @ W_key
values = inputs @ W_value

print(keys)
print(values)

keys_2 = keys[1]
attn_score_22 = torch.dot(query_2, keys_2)
print(attn_score_22)