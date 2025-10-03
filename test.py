################################################################################
# filename: test.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 16/07,2025
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

B,T,C = 4,8,32
x= torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False) # self attention because it's x
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False) # self attention because it's x

k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)
wei = q @ k.transpose(-2, -1) * head_size ** -0.5 # (B, T, 16) @ (B, 16, T) / sqrt(dk) (ramene à 1) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ value(x)

print(out.shape)