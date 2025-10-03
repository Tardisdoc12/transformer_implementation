
import torch
from torch import nn
from torch.nn import functional as F

from encoder import EncoderLargeModel
from decoder import BigramLanguageModel

vocab_size = 65
block_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"


class ChatBot(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderLargeModel()
        self.decoder = BigramLanguageModel(vocab_size)

    def forward(self, idx, targets=None):
        x = self.encoder(idx, targets)
        logits, loss = self.decoder(idx, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
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
        return self.decoder.generate(idx, max_new_tokens)
    
