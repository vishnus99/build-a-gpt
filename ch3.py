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


The goal of causal attention is to compute a context vector for each input element that combines information from all other input elements before the current element.
This is used in decoder only models like GPT.
Masking + dropout is used to prevent the model from using information from future tokens and to prevent overfitting. 


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
'''
Note about the size of the dimensions:
1. Input matrix dimensions are determined by the sequence length(number of tokens) and the embedding dimension(size of each token's representation)
2. Weight matrix dimensions have the same input and output dimensions as the embedding dimension, this can vary.
3. The output matrix dimensions are determined by the sequence length and the input embedding dimension.


Multi-head attention:

Input: (1, 1024, 768)
    ↓
Split into 12 heads:
Head 1: (1, 1024, 64) ← Q1, K1, V1
Head 2: (1, 1024, 64) ← Q2, K2, V2
...
Head 12: (1, 1024, 64) ← Q12, K12, V12
    ↓
Each head computes attention independently
    ↓
Concatenate: (1, 1024, 768)
    ↓
Final projection: (1, 1024, 768)


'''

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
'''
Note about initializing the weights:

What activation function are you using?
├── Sigmoid/Tanh → Xavier/Glorot
├── ReLU/Leaky ReLU → Kaiming/He
└── Linear/No activation → Normal (small std)

What type of network?
├── Traditional MLP with sigmoid → Xavier/Glorot
├── CNN with ReLU → Kaiming/He
├── Transformer/Attention → Normal (0.02 std)
└── RNN/LSTM → Xavier/Glorot (for sigmoid gates)

'''
query_2 = x_2 @ W_query

print(query_2)

keys = inputs @ W_key
values = inputs @ W_value

print(keys)
print(values)

keys_2 = keys[1]
attn_score_22 = torch.dot(query_2, keys_2)
print(attn_score_22)

'''
Pseudocode for self-attention (with trainable weights):

1. Compute the query, key, and value vectors for each input element using matrix multiplication between the input token and the trainable weight matrices.
2. Compute the attention weights for each input element with respect to all other input elements using the dot product of the query vector and the key vector.
3. Normalize the attention weights by applying a softmax function.
4. Multiply the value vectors by the now normalized attention weights and sum them up to get the context vector for the input element.

'''

'''
Pseudocode for causal attention (with trainable weights):

1. Compute the query, key, and value vectors for each input element using matrix multiplication between the input token and the trainable weight matrices.
2. Compute the attention weights for each input element with respect to all other input elements using the dot product of the query vector and the key vector.
3. Normalize the attention weights by applying a softmax function.
4. Mask the attention weights to prevent the model from using information from future tokens.
5. Multiply the value vectors by the now normalized attention weights and sum them up to get the context vector for the input element.

'''

'''
Pseudocode for multi-head attention (with trainable weights):

1. Compute the query, key, and value vectors for each input element using matrix multiplication between the input token and the trainable weight matrices.
2. Split the query, key, and value vectors into multiple heads.
3. Compute the attention weights for each head using the dot product of the query vector and the key vector.
4. Normalize the attention weights by applying a softmax function.
5. Multiply the value vectors by the now normalized attention weights and sum them up to get the context vector for the input element.
6. Concatenate the context vectors from each head and project them back to the original embedding dimension.

'''


#Multi-head attention class
torch.nn.MultiheadAttention(embed_dim = d_in, num_heads = 12, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)