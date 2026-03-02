import json
import os
from datetime import datetime

import torch
from tqdm import tqdm


def levenshtein(a, b):
    """
    Token-level Levenshtein distance
    a, b: list of tokens (or strings split into tokens)
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[n][m]


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    tokenizer,
    device,
    decode_fn,
    max_len=150,
    anomaly_log_path="logs/val_anomaly_samples.jsonl",
):
    model.eval()

    total = 0
    exact_match = 0
    total_edit_distance = 0
    total_ref_len = 0

    skipped_batches = 0
    os.makedirs(os.path.dirname(anomaly_log_path), exist_ok=True)

    with open(anomaly_log_path, "a", encoding="utf-8") as log_f:
        for batch_idx, (images, gt_ids, sample_paths) in enumerate(
            tqdm(dataloader, desc="Evaluating")
        ):

            images = images.to(device)
            gt_ids = gt_ids.to(device)

            tf_logits = model(images, gt_ids[:, :-1])
            invalid_mask = torch.isnan(tf_logits) | torch.isinf(tf_logits)
            if invalid_mask.any():
                skipped_batches += 1
                anomaly_record = {
                    "time": datetime.now().isoformat(),
                    "type": "invalid_logits",
                    "batch_idx": batch_idx,
                    "sample_paths": list(sample_paths),
                }
                log_f.write(json.dumps(anomaly_record, ensure_ascii=False) + "\n")
                print(
                    f"[WARN] Skip batch {batch_idx}: NaN/Inf logits detected. "
                    f"Samples={list(sample_paths)}"
                )
                continue

            batch_size = images.size(0)

            for i in range(batch_size):

                single_image = images[i:i + 1]
                pred_str, pred_ids = decode_fn(
                    model,
                    single_image,
                    tokenizer,
                    max_len=max_len,
                    return_ids=True,
                )

                gt_token_ids = gt_ids[i].tolist()
                gt_str = tokenizer.decode(gt_token_ids)

                pred_str = pred_str.strip()
                gt_str = gt_str.strip()[1:-1]

                if total < 5:
                    print("\nGT  :", gt_str)
                    print("PRED:", pred_str)
                    print("-" * 40)

                if pred_str == gt_str:
                    exact_match += 1

                pred_tokens = tokenizer.tokenize(pred_str)
                gt_tokens = tokenizer.tokenize(gt_str)

                ed = levenshtein(pred_tokens, gt_tokens)

                total_edit_distance += ed
                total_ref_len += len(gt_tokens)
                total += 1

                sample_record = {
                    "time": datetime.now().isoformat(),
                    "type": "decode_sample",
                    "batch_idx": batch_idx,
                    "sample_path": sample_paths[i],
                    "gt_token_ids": gt_token_ids,
                    "pred_token_ids": pred_ids,
                    "gt_text": gt_str,
                    "pred_text": pred_str,
                }
                log_f.write(json.dumps(sample_record, ensure_ascii=False) + "\n")

    if total == 0:
        return {
            "ExactMatch": 0.0,
            "AvgEditDistance": 0.0,
            "NormEditDistance": 0.0,
            "SkippedBatches": skipped_batches,
        }

    return {
        "ExactMatch": exact_match / total,
        "AvgEditDistance": total_edit_distance / total,
        "NormEditDistance": total_edit_distance / max(total_ref_len, 1),
        "SkippedBatches": skipped_batches,
    }
