from GPT_model_script.GPT import GPTModel, generate_text_simple
import torch
import tiktoken
import os
from torch.utils.data import Dataset, DataLoader, random_split

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

#print(text_data[:50])
test = tokenizer.encode(text_data, allowed_special={"<|endoftext|>"})
test_tensor = torch.tensor(test).unsqueeze(0)
#print(test_tensor)
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

#print(f"Total characters: {total_characters}")
#print(f"Total tokens: {total_tokens}")

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
    

def collate_fn(batch, batch_size, context_length):
    original_batch_length = len(batch)
    mask = torch.ones(batch_size, context_length)
    if original_batch_length < batch_size:
          padding_tensor = torch.zeros(context_length, dtype=batch[0].dtype)
          pads_to_add = batch_size - original_batch_length
          batch.extend([padding_tensor] * pads_to_add)
          stacked_batch = torch.stack(batch)
          mask[original_batch_length:batch_size, :] = 0
    else:
         stacked_batch = torch.stack(batch)
    return stacked_batch, mask



#Testing dataset and dataloader
# dataset = TextDataset(text_data, tokenizer, GPT_CONFIG_124M["context_length"])
# dataloader = DataLoader(dataset, 
#                         batch_size=2, 
#                         collate_fn=lambda batch: collate_fn(batch, batch_size=2, context_length=GPT_CONFIG_124M["context_length"])
#                         )

# for i, (batch_tensor, attention_mask) in enumerate(dataloader):
#      print(f"Batch: {i}: ")
#      print(f"   Tensor shape: {batch_tensor.shape}")
#      print(f"   Mask shape {attention_mask.shape}")
#      print(f"   Mask values: {attention_mask}")
#      print(f"   First sequence tokens: {batch_tensor[0][:10]}...")

#      if i >=2: 
#           break

torch.manual_seed(123)

dataset = TextDataset(text_data, tokenizer, GPT_CONFIG_124M["context_length"])

train_to_val_ratio = 0.90
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

#print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

train_dataloader = DataLoader(train_data, 
                              batch_size=2, 
                              collate_fn=lambda batch: collate_fn(batch, batch_size=2, context_length=GPT_CONFIG_124M["context_length"])
                              )

val_dataloader = DataLoader(val_data, 
                              batch_size=2, 
                              collate_fn=lambda batch: collate_fn(batch, batch_size=2, context_length=GPT_CONFIG_124M["context_length"])
                              )

#Loss function
def calc_loss_batch(input_batch, attention_mask, model, device):
     input_batch = input_batch.to(device)
     target_batch = input_batch[:, 1:]
     target_batch = target_batch.to(device)
     attention_mask = attention_mask.to(device)
     attention_mask = attention_mask[:, 1:]
     logits = model(input_batch)
     logits = logits[:, :-1, :]
     loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten(), reduction='none')
     masked_loss = loss * attention_mask.flatten()
     final_loss = masked_loss.sum() / attention_mask.sum()
     return final_loss



#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)


#Training function
def train_simple_model(model, train_dataloader, val_dataloader, num_epochs, optimizer, device):
     for epoch in range(num_epochs):
          model.train()
          epoch_loss = 0
          for batch_idx, (batch, mask) in enumerate(train_dataloader):
               optimizer.zero_grad()
               loss = calc_loss_batch(batch, mask, model, device)
               loss.backward()
               optimizer.step()
               epoch_loss += loss.item()
          avg_loss = epoch_loss / len(train_dataloader)
          print(f"Epoch {epoch}: Average loss = {avg_loss}")
          
          model.eval()
          val_epoch_loss = 0
          with torch.no_grad():
               for batch, mask in val_dataloader:
                    val_loss = calc_loss_batch(batch, mask, model, device)
                    val_epoch_loss += val_loss.item()
          avg_val_loss = val_epoch_loss / len(val_dataloader)         
          print(f"Average validation loss = {avg_val_loss}")
          model.train()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_simple_model(
     model=model,
     train_dataloader=train_dataloader,
     val_dataloader=val_dataloader,
     num_epochs=20,
     optimizer=optimizer,
     device=device
)




    


    
