import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model.im2latex import Im2Latex
from tokenizer import LatexTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def beam_search_decode(
    model,
    image,
    tokenizer,
    beam_size=5,
    max_len=120,
    length_penalty=0.8,
    repeat_penalty=1.2,
    topk_per_beam=32,
):
    device = image.device
    sos = tokenizer.token2id["<SOS>"]
    eos = tokenizer.token2id["<EOS>"]

    banned_tokens = {
        tokenizer.token2id.get("<PAD>"),
        tokenizer.token2id.get("<SOS>"),
        tokenizer.token2id.get("<UNK>"),
    }

    beams = [([sos], 0.0, False)]

    for _ in range(max_len):
        new_beams = []

        for seq, score, finished in beams:
            if finished:
                new_beams.append((seq, score, True))
                continue

            tgt = torch.tensor([seq], device=device)

            with torch.no_grad():
                logits = model(image, tgt)[:, -1]
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

            k = min(topk_per_beam, log_probs.numel())
            top_scores, top_ids = torch.topk(log_probs, k=k)

            for token_score, token_id_t in zip(top_scores.tolist(), top_ids.tolist()):
                token_id = int(token_id_t)

                if token_id in banned_tokens:
                    continue

                # Correct repetition penalty on log-probabilities.
                if len(seq) >= 1 and token_id == seq[-1]:
                    token_score *= repeat_penalty

                finished_flag = token_id == eos
                new_beams.append((seq + [token_id], score + token_score, finished_flag))

        if not new_beams:
            break

        beams = sorted(
            new_beams,
            key=lambda x: x[1] / (len(x[0]) ** length_penalty),
            reverse=True,
        )[:beam_size]

        if all(f for _, _, f in beams):
            break

    best_seq = beams[0][0] if beams else [sos, eos]
    return tokenizer.decode(best_seq)


def infer(img_path):
    tokenizer = LatexTokenizer()
    tokenizer.load_vocab("vocab.json")

    model = Im2Latex(len(tokenizer.token2id), pad_idx=tokenizer.token2id["<PAD>"]).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Image load failed."

    img = cv2.resize(img, (256, 64))
    img = img.astype(np.float32) / 255.0
    img = np.stack([img, img, img], axis=0)

    img = torch.tensor(img).unsqueeze(0).to(device)

    return beam_search_decode(model, img, tokenizer)
