from GPT_model_script.GPT import GPTModel, generate_text_simple
import torch
import tiktoken
import os
from torch.utils.data import Dataset, DataLoader

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 1024, 
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

def text_to_tokens(text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

def tokens_to_text(tokens, tokenizer):
    flatten = tokens.squeeze(0)
    decoded = tokenizer.decode(flatten.tolist())
    return decoded

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(model, text_to_tokens(start_context, tokenizer), 10, GPT_CONFIG_124M["context_length"])

file_path = "../build-a-gpt/scraper_and_sample_text/hltv_forum_threads.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

print(text_data[:50])
test = tokenizer.encode(text_data, allowed_special={"<|endoftext|>"})
test_tensor = torch.tensor(test).unsqueeze(0)
print(test_tensor)
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print(f"Total characters: {total_characters}")
print(f"Total tokens: {total_tokens}")

class TextDataset(Dataset):
    def __init__(self, text_data, tokenizer, context_length):
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.context_length = context_length

        self.tokens = torch.tensor(tokenizer.encode(text_data, allowed_special = {"<|endoftext|>"})).unsqueeze(0)
    def __len__(self):
         return self.tokens.shape[1]//self.context_length
    
    def __getitem__(self, idx):
        return self.tokens[0, idx*self.context_length:(idx+1)*self.context_length]
    
dataset = TextDataset(text_data=text_data, tokenizer=tokenizer, context_length=GPT_CONFIG_124M["context_length"])

print(f"Dataset length: {len(dataset)}")
print(f"First sequence shape: {dataset[0].shape}")        
print(f"First sequence: {dataset[0]}")

def collate_fn(batch, batch_size):
    if len(batch) < batch_size:
          padding_tensor = torch.zeros(1024)
          pads_to_add = batch_size - len(batch)
          batch.extend([padding_tensor] * pads_to_add)
          stacked_batch = torch.stack(batch)
    else:
         stacked_batch = torch.stack(batch)
    return stacked_batch

dataloader = DataLoader(dataset, batch_size = 2, shuffle=True, collate_fn=collate_fn)





    


    
