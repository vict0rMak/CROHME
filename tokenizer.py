# tokenizer.py
import json
import string


class LatexTokenizer:
    # def __init__(self):
    #     self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    #
    #     self.structure_tokens = [
    #         r"\frac", r"\sqrt",
    #         r"\left(", r"\right)",
    #         r"\left[", r"\right]",
    #         r"\left\{", r"\right\}",
    #         r"^{", r"}", r"_{", r"}",
    #     ]
    #
    #     self.greek_letters = [
    #         r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
    #         r"\zeta", r"\eta", r"\theta", r"\iota", r"\kappa",
    #         r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi",
    #         r"\rho", r"\sigma", r"\tau", r"\upsilon",
    #         r"\phi", r"\chi", r"\psi", r"\omega",
    #         r"\Alpha", r"\Beta", r"\Gamma", r"\Delta", r"\Epsilon",
    #         r"\Zeta", r"\Eta", r"\Theta", r"\Iota", r"\Kappa",
    #         r"\Lambda", r"\Mu", r"\Nu", r"\Xi", r"\Pi",
    #         r"\Rho", r"\Sigma", r"\Tau", r"\Upsilon",
    #         r"\Phi", r"\Chi", r"\Psi", r"\Omega",
    #     ]
    #
    #     self.hebrew_letters = [
    #         r"\aleph", r"\beth", r"\gimel", r"\daleth"
    #     ]
    #
    #     self.operators = [
    #         "+", "-", "*", "/", "=",
    #         r"\cdot", r"\times",
    #     ]
    #
    #     self.brackets = ["(", ")", "[", "]", "{", "}"]
    #
    #     self.letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    #     self.digits = list("0123456789")
    #
    #     # 优先级：长 token 在前
    #     self.all_tokens = (
    #         self.special_tokens +
    #         self.structure_tokens +
    #         self.greek_letters +
    #         self.hebrew_letters +
    #         self.operators +
    #         self.brackets +
    #         self.letters +
    #         self.digits
    #     )
    #
    #     self.token2id = {t: i for i, t in enumerate(self.all_tokens)}
    #     self.id2token = {i: t for t, i in self.token2id.items()}

    def __init__(self):
        self.special_tokens = [
            "<PAD>", "<SOS>", "<EOS>", "<UNK>"
        ]

        self.latex_tokens = [
            # Structural commands
            r"\frac", r"\sqrt", r"\sum", r"\int", r"\lim",
            r"\log", r"\ln", r"\exp",
            r"\sin", r"\cos", r"\tan",

            # Scripts
            r"^{", r"_{", r"}",

            # Brackets
            r"\left(", r"\right)",
            r"\left[", r"\right]",
            r"\left\{", r"\right\}",
            r"\left|", r"\right|",
            "(", ")", "[", "]", "{", "}", "|",

            # Operators
            "+", "-", "*", "/", "=",
            "<", ">", r"\le", r"\ge", r"\neq", r"\approx",

            # Greek letters
            r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
            r"\zeta", r"\eta", r"\theta", r"\iota", r"\kappa",
            r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi",
            r"\rho", r"\sigma", r"\tau", r"\upsilon",
            r"\phi", r"\chi", r"\psi", r"\omega",
            r"\Alpha", r"\Beta", r"\Gamma", r"\Delta", r"\Epsilon",
            r"\Zeta", r"\Eta", r"\Theta", r"\Iota", r"\Kappa",
            r"\Lambda", r"\Mu", r"\Nu", r"\Xi", r"\Pi",
            r"\Rho", r"\Sigma", r"\Tau", r"\Upsilon",
            r"\Phi", r"\Chi", r"\Psi", r"\Omega",
            # Hebrew letters
            r"\aleph", r"\beth", r"\gimel", r"\daleth",
            r"\cdot", r"\times",
        ]

        # Characters
        self.char_tokens = list(string.ascii_letters + string.digits + ".,:")

        self.token2id = {}
        self.id2token = {}
        self._sorted_latex_tokens = sorted(set(self.latex_tokens), key=len, reverse=True)

    def build_vocab(self, json_paths):
        vocab = set(self.latex_tokens + self.char_tokens)

        for path in json_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    vocab.update(self.tokenize(item["latex"]))

        all_tokens = self.special_tokens + sorted(vocab, key=lambda x: (-len(x), x))
        self.token2id = {t: i for i, t in enumerate(all_tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}

    def tokenize(self, latex):
        tokens = []
        i = 0
        while i < len(latex):
            matched = False
            for tok in self._sorted_latex_tokens:
                if latex.startswith(tok, i):
                    tokens.append(tok)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                tokens.append(latex[i])
                i += 1
        return tokens

    def encode(self, latex, max_len=150):
        tokens = ["<SOS>"] + self.tokenize(latex) + ["<EOS>"]
        ids = [self.token2id.get(t, self.token2id["<UNK>"]) for t in tokens]
        ids = ids[:max_len]
        if ids and ids[-1] != self.token2id["<EOS>"]:
            ids[-1] = self.token2id["<EOS>"]
        ids += [self.token2id["<PAD>"]] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            tok = self.id2token.get(i, "")
            if tok == "<EOS>":
                break
            if tok not in self.special_tokens:
                tokens.append(tok)
        return "".join(tokens)

    def save_vocab(self, path="vocab.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path="vocab.json"):
        with open(path, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)
        self.id2token = {i: t for t, i in self.token2id.items()}
