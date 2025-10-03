################################################################################
# filename: embedding_image.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 19/07,2025
################################################################################

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
import cv2

import os

################################################################################

n_embed = 32
patch_size = 16
n_heads = 4
n_block = 6
dropout = 0.1

################################################################################

class EmbeddingImage(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=3):
        super().__init__()
        self.img_size = img_size      # (H, W)
        self.patch_size = patch_size  # (P, P)
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.patch_dim = in_channels * patch_size * patch_size
        # Embedding d'un patch à un vecteur de dimension n_embed
        self.linear_projection = nn.Linear(self.patch_dim, n_embed)

        # Encodage de position appris
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, n_embed))

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        x = x.unfold(2, P, P).unfold(3, P, P)
        x = x.contiguous().view(B, C, -1, P, P)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B,self.n_patches, -1)

        x = self.linear_projection(x) + self.pos_embedding
        return x

################################################################################

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, hs) @ (B, hs, T) / sqrt(dk) (ramene à 1) ---> (B, T, T)
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

class EncoderImageTransformers(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingImage(img_size=(256, 256), patch_size=16)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_block)])
        self.ln = nn.LayerNorm(n_embed)
    
    def forward(self, img):
        x = self.embedding(img)
        x = self.blocks(x)
        return self.ln(x)
    
################################################################################

class Classificator(nn.Module):
    def __init__(self):
        super().__init__()
        self.classificator = nn.Linear(n_embed, 3)
    
    def forward(self, x):
        x = x.mean(dim=1) 
        return self.classificator(x)
    
################################################################################

class RPSDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.samples = []
        self.class_to_idx = {"paper": 0, "rock": 1, "scissors": 2}
        for label, files in data_dict.items():
            for filepath in files:
                self.samples.append((filepath, self.class_to_idx[label]))

        self.transform = transform or T.Compose([
            T.ToTensor(),  # cv2 image is in np.ndarray format
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         # BGR → RGB
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = self.transform(img)                          # Convert to tensor
        return img, label

################################################################################

@torch.no_grad()
def estimate_loss(model, test_loader, device="cuda"):
    model.eval()
    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        outputs = classificator(outputs)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
    model.train()
    return correct / len(test_loader.dataset)

################################################################################

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    import torch.optim as optim

    model = EncoderImageTransformers()
    classificator = Classificator()

    paper_scissors_rock_dataset = {
        "paper": ["paper/" + file for file in os.listdir("paper/")],
        "rock": ["rock/" + file for file in os.listdir("rock/")],
        "scissors": ["scissors/" + file for file in os.listdir("scissors/")],
    }

    train_dict, test_dict = {}, {}
    for label, files in paper_scissors_rock_dataset.items():
        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
        train_dict[label] = train_files
        test_dict[label] = test_files
    
    train_dataset = RPSDataset(train_dict)
    test_dataset = RPSDataset(test_dict)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load("model.pth"))
    classificator.load_state_dict(torch.load("classificator.pth"))
    classificator = classificator.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # for epoch in range(100):
    #     model.train()
    #     if epoch % 10 == 0 and epoch != 0:
    #         accuracy = estimate_loss(model, test_dataloader, device=device)
    #         if accuracy > 0.9:
    #             torch.save(model.state_dict(), "model.pth")
    #             break
    #         else:
    #             print(f"Accuracy: {accuracy:.4f}")
    #     total_loss = 0
    #     correct = 0
    #     for images, labels in train_dataloader:
    #         images, labels = images.to(device), labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         outputs = classificator(outputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()
    #         preds = outputs.argmax(dim=1)
    #         correct += (preds == labels).sum().item()
    #         torch.save(model.state_dict(), "model.pth")
    #         torch.save(classificator.state_dict(), "classificator.pth")
    #     print(f"Epoch: {epoch}, Loss: {total_loss / len(train_dataloader):.4f}, Accuracy: {correct / len(train_dataloader.dataset):.4f}")

    # on essaie avec l'image pierre.png
    image = cv2.imread("ciseaux.png")
    transform = T.Compose([
            T.ToTensor(),  # cv2 image is in np.ndarray format
        ])
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)         # BGR → RGB
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    output = model(img)
    logits = classificator(output)

    probas = logits.softmax(dim=1)
    print(probas.argmax(dim=1))
