################################################################################
# filename: translator_code.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 16/07,2025
################################################################################
# IMPORTS

import os
from typing import *

import torch
import torch.nn as nn
from torch.nn import functional as F

################################################################################
# Constantes:

torch.manual_seed(1337)

batch_size = 64
block_size = 256
max_iter = 5_000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

################################################################################

def read_data(file_path: str) -> str:
    with open(file_path, 'r',encoding='utf-8') as f:
        data = f.read()
    return data

################################################################################

def get_voabulary(data : str) -> tuple[list, int]:
    vocab = sorted(list(set(data)))
    return vocab, len(vocab)

################################################################################

def get_mapping_char_to_int(vocab : list) -> dict:
    stoi = {char: i for i, char in enumerate(vocab)}
    itos = {i: char for i, char in enumerate(vocab)}
    return stoi, itos

################################################################################

def get_encoder_decoder(stoi : dict, itos : dict) -> tuple[Callable, Callable]: # tiktoken pour gpt
    encoder = lambda x: [stoi[char] for char in x]
    decoder = lambda x: [itos[char] for char in x]
    return encoder, decoder

################################################################################

def get_tensor(text_to_encode : str, encoder : Callable) -> torch.Tensor:
    return torch.tensor(encoder(text_to_encode), dtype=torch.long)

################################################################################

def split_train_val(data : torch.Tensor, train_size : float) -> tuple[torch.Tensor, torch.Tensor]:
    train_size = int(train_size * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    return train_data, val_data

################################################################################

def get_block_size_data(data : torch.Tensor, block_size : int) -> int:
    return data[:block_size + 1]

################################################################################

def get_batch(data, block_size,batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

################################################################################

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, hs) @ (B, hs, T) / sqrt(dk) (ramene à 1) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) ---> (B, T, hs)
        return out

################################################################################

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

################################################################################

class feedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

################################################################################

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = feedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

################################################################################

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embeding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embeding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
        return idx

################################################################################

def get_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)

################################################################################

@torch.no_grad()
def estimate_loss(model, train_data, eval_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            dataset = train_data if split == 'train' else eval_data
            X, Y = get_batch(dataset, block_size, batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

################################################################################

if __name__ == "__main__":
    # Exécution du code
    datas = read_data("input.txt")
    vocab, vocab_size = get_voabulary(datas)
    stoi, itos = get_mapping_char_to_int(vocab)
    encoder, decoder = get_encoder_decoder(stoi, itos)
    tensor_text_shake = get_tensor(datas, encoder)
    train_dataset, val_dataset = split_train_val(tensor_text_shake, 0.9)
    xb, yb = get_batch(train_dataset, 8, 4)
    m = BigramLanguageModel(vocab_size).to(device)
    logits, loss = m(xb, yb)
    optimizer = get_optimizer(m)
    old_loss = 100
    os.mkdir("models")
    for steps in range (max_iter):

        if steps % eval_interval == 0:
            losses = estimate_loss(m, train_dataset, val_dataset)
            torch.save(m.state_dict(), "models/model.pth")
            if losses['val'] < old_loss:
                torch.save(m.state_dict(), "models/best_model.pth")
                old_loss = losses['val']
            print(f"Step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(train_dataset, block_size, batch_size)
        
        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print(loss.item())
    print("".join(decoder(m.generate(torch.zeros((1, 1), dtype=torch.long, device = device), max_new_tokens=500)[0].tolist())))

################################################################################
# End of File
################################################################################