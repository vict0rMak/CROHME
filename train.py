from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CROHMEDataset
from evaluation import evaluate
from infer import beam_search_decode
from model.im2latex import Im2Latex
from tokenizer import LatexTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"

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
model = Im2Latex(len(tokenizer.token2id), pad_idx=tokenizer.token2id["<PAD>"]).to(device)

criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.token2id["<PAD>"],
    label_smoothing=0.1
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
num_epochs = 60
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=8e-5)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

patience = 8
best_norm_ed = float("inf")
early_stop_counter = 0

# ===============================
# 4️⃣ Training Loop
# ===============================
for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch} =====")
    model.train()

    total_loss = 0.0
    correct, total = 0, 0

    for img, tgt in tqdm(train_loader, desc="Training"):
        img, tgt = img.to(device), tgt.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(img, tgt[:, :-1])
            loss = criterion(
                out.reshape(-1, out.size(-1)),
                tgt[:, 1:].reshape(-1)
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # token accuracy (monitor only)
        pred = out.argmax(-1)
        gt = tgt[:, 1:]
        mask = gt != tokenizer.token2id["<PAD>"]
        correct += ((pred == gt) & mask).sum().item()
        total += mask.sum().item()

    scheduler.step()

    train_loss = total_loss / len(train_loader)
    token_acc = correct / max(1, total)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train TokenAcc: {token_acc:.4f}")
    print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # ===============================
    # 5️⃣ Validation (EM + NormED)
    # ===============================
    decode_fn = partial(
        beam_search_decode,
        beam_size=4,
        max_len=120,
        length_penalty=0.75,
        repeat_penalty=1.2,
        topk_per_beam=24,
    )

    metrics = evaluate(
        model=model,
        dataloader=val_loader,
        tokenizer=tokenizer,
        device=device,
        decode_fn=decode_fn,
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
