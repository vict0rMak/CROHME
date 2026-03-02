import torch
from torch.utils.data import DataLoader
from dataset import CROHMEDataset
from tokenizer import LatexTokenizer
from model.im2latex import Im2Latex
import torch.nn as nn
from tqdm import tqdm
from evaluation import evaluate
from infer import beam_search_decode

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 1️⃣ Tokenizer
# ===============================
tokenizer = LatexTokenizer()
tokenizer.build_vocab([
    "data/processed/train.json",
    "data/processed/val.json"
])
tokenizer.save_vocab("vocab.json")

# ===============================
# 2️⃣ Dataset
# ===============================
train_ds = CROHMEDataset(
    "data/processed/train.json",
    "data/processed/images",
    tokenizer
)

val_ds = CROHMEDataset(
    "data/processed/val.json",
    "data/processed/images",
    tokenizer
)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# ===============================
# 3️⃣ Model
# ===============================

model = Im2Latex(len(tokenizer.token2id)).to(device)

criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.token2id["<PAD>"],
    label_smoothing=0.1
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

num_epochs = 60
patience = 8
best_norm_ed = 114514.0
early_stop_counter = 0

# ===============================
# 4️⃣ Training Loop
# ===============================

for epoch in range(num_epochs):

    print(f"\n===== Epoch {epoch} =====")
    model.train()

    total_loss = 0
    correct, total = 0, 0

    for img, tgt in tqdm(train_loader, desc="Training"):
        img, tgt = img.to(device), tgt.to(device)

        out = model(img, tgt[:, :-1])

        loss = criterion(
            out.reshape(-1, out.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

        # token accuracy (仅监控)
        pred = out.argmax(-1)
        gt = tgt[:, 1:]
        mask = gt != tokenizer.token2id["<PAD>"]
        correct += ((pred == gt) & mask).sum().item()
        total += mask.sum().item()

    train_loss = total_loss / len(train_loader)
    token_acc = correct / total

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train TokenAcc: {token_acc:.4f}")

    # ===============================
    # 5️⃣ Validation (EM + NormED)
    # ===============================

    metrics = evaluate(
        model=model,
        dataloader=val_loader,
        tokenizer=tokenizer,
        device=device,
        decode_fn=beam_search_decode
    )

    val_em = metrics["ExactMatch"]
    val_norm_ed = metrics["NormEditDistance"]

    print(f"Val EM: {val_em:.4f}")
    print(f"Val NormED: {val_norm_ed:.4f}")

    # ===============================
    # 6️⃣ Early Stopping
    # ===============================

    if val_norm_ed < best_norm_ed:
        print("New Best Model Saved.")
        best_norm_ed = val_norm_ed
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        early_stop_counter += 1
        print(f"No improvement. Patience {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

print("\nTraining Finished.")
print(f"Best Val NormED: {best_norm_ed:.4f}")
