"""
String Reversal Dataset
"abcde" -> "edcba"
"""

import random
import string

import torch
from torch.utils.data import Dataset

from src.vocab import build_vocab, encode

CHARS = list(string.ascii_lowercase + string.digits)
char2idx, idx2char, PAD_IDX, BOS_IDX, EOS_IDX = build_vocab(CHARS)
VOCAB_SIZE = len(char2idx)


def _random_string(min_len: int, max_len: int) -> str:
    length = random.randint(min_len, max_len)
    return "".join(random.choices(CHARS, k=length))


class ReverseStringDataset(Dataset):
    def __init__(self, size: int, min_len: int = 3, max_len: int = 10):
        self.data = []
        for _ in range(size):
            src = _random_string(min_len, max_len)
            tgt = src[::-1]
            self.data.append((
                encode(src, char2idx),
                [BOS_IDX] + encode(tgt, char2idx),
                encode(tgt, char2idx) + [EOS_IDX],
            ))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    src_list, tgt_in_list, tgt_out_list = zip(*batch)

    def pad(seqs):
        max_len = max(len(s) for s in seqs)
        return torch.tensor(
            [s + [PAD_IDX] * (max_len - len(s)) for s in seqs],
            dtype=torch.long,
        )

    return pad(src_list), pad(tgt_in_list), pad(tgt_out_list)