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
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
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
):
    model.eval()

    total = 0
    exact_match = 0
    total_edit_distance = 0
    total_ref_len = 0

    for images, gt_ids in tqdm(dataloader, desc="Evaluating"):

        images = images.to(device)
        gt_ids = gt_ids.to(device)

        batch_size = images.size(0)

        for i in range(batch_size):

            single_image = images[i:i+1]  # 更安全写法

            pred_str = decode_fn(
                model,
                single_image,
                tokenizer
            )

            gt_str = tokenizer.decode(gt_ids[i].tolist())

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

    return {
        "ExactMatch": exact_match / total,
        "AvgEditDistance": total_edit_distance / total,
        "NormEditDistance": total_edit_distance / total_ref_len
    }
