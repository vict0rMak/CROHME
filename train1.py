import torch
from torch.utils.data import DataLoader
from dataset import CROHMEDataset
from tokenizer import LatexTokenizer
from model.im2latex import Im2Latex
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = LatexTokenizer()
tokenizer.build_vocab([
    "data/processed/train.json",
    "data/processed/val.json"
])
tokenizer.save_vocab("vocab.json")

train_ds = CROHMEDataset(
    "data/processed/train.json",
    "data/processed/images",
    tokenizer
)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

model = Im2Latex(len(tokenizer.token2id)).to(device)

criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.token2id["<PAD>"],
    label_smoothing=0.1
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
num_epochs = 80

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for img, tgt in train_loader:
        img, tgt = img.to(device), tgt.to(device)
        out = model(img, tgt)

        loss = criterion(
            out.reshape(-1, out.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = out.argmax(-1)
        gt = tgt[:, 1:]
        mask = gt != tokenizer.token2id["<PAD>"]
        correct += ((pred == gt) & mask).sum().item()
        total += mask.sum().item()

    print(
        f"Epoch {epoch} | "
        f"Loss {total_loss / len(train_loader):.4f} | "
        f"TokenAcc {correct / total:.4f}"
    )

torch.save(model.state_dict(), "model.pth")
