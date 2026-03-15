"""
Integer Addition Dataset
"123+456" -> "579"
"999+1"   -> "1000"  (carry!)
"""

import random

import torch
from torch.utils.data import Dataset

from src.vocab import build_vocab, encode

CHARS = list("0123456789+")
char2idx, idx2char, PAD_git IDX, BOS_IDX, EOS_IDX = build_vocab(CHARS)
VOCAB_SIZE = len(char2idx)


def _random_sample(max_digits: int) -> tuple[str, str]:
    max_val = 10 ** max_digits - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    return f"{a}+{b}", str(a + b)


class AdditionDataset(Dataset):
    def __init__(self, size: int, max_digits: int):
        self.max_digits = max_digits
        self.data = []
        seen: set[str] = set()
        while len(self.data) < size:
            src, tgt = _random_sample(max_digits)
            if src in seen:
                continue
            seen.add(src)
            self.data.append((
                encode(src, char2idx),
                [BOS_IDX] + encode(tgt, char2idx),
                encode(tgt, char2idx) + [EOS_IDX],
                src, tgt,
            ))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_in, tgt_out, _, _ = self.data[idx]
        return src_ids, tgt_in, tgt_out

    def raw(self, idx) -> tuple[str, str]:
        return self.data[idx][3], self.data[idx][4]


def collate_fn(batch):
    src_list, tgt_in_list, tgt_out_list = zip(*batch)

    def pad(seqs):
        max_len = max(len(s) for s in seqs)
        return torch.tensor(
            [s + [PAD_IDX] * (max_len - len(s)) for s in seqs],
            dtype=torch.long,
        )

    return pad(src_list), pad(tgt_in_list), pad(tgt_out_list)