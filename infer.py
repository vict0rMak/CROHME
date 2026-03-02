import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tokenizer import LatexTokenizer
from model.im2latex import Im2Latex

device = "cuda" if torch.cuda.is_available() else "cpu"


def beam_search_decode(
    model,
    image,
    tokenizer,
    beam_size=5,
    max_len=150,
    length_penalty=0.8,
    repeat_penalty=1.1
):
    device = image.device
    sos = tokenizer.token2id["<SOS>"]
    eos = tokenizer.token2id["<EOS>"]

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
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

            for token_id in range(len(log_probs)):
                token_score = log_probs[token_id].item()

                # soft repetition penalty
                if len(seq) >= 1 and token_id == seq[-1]:
                    token_score /= repeat_penalty

                finished_flag = (token_id == eos)

                new_beams.append(
                    (seq + [token_id],
                     score + token_score,
                     finished_flag)
                )

        beams = sorted(
            new_beams,
            key=lambda x: x[1] / (len(x[0]) ** length_penalty),
            reverse=True
        )[:beam_size]

        if all(f for _, _, f in beams):
            break

    best_seq = beams[0][0]
    return tokenizer.decode(best_seq)


def infer(img_path):
    tokenizer = LatexTokenizer()
    tokenizer.load_vocab("vocab.json")

    model = Im2Latex(len(tokenizer.token2id)).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Image load failed."

    img = cv2.resize(img, (256, 64))
    img = img.astype(np.float32) / 255.0
    img = np.stack([img, img, img], axis=0)

    img = torch.tensor(img).unsqueeze(0).to(device)

    return beam_search_decode(
        model,
        img,
        tokenizer
    )
