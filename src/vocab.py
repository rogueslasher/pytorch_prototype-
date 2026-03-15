
PAD, BOS, EOS = "<PAD>", "<BOS>", "<EOS>"


def build_vocab(chars: list[str]) -> tuple[dict, dict, int, int, int]:
    vocab = [PAD, BOS, EOS] + chars
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char, char2idx[PAD], char2idx[BOS], char2idx[EOS]


def encode(s: str, char2idx: dict) -> list[int]:
    return [char2idx[c] for c in s]


def decode(ids: list[int], idx2char: dict,
           pad_idx: int, bos_idx: int, eos_idx: int) -> str:
    out = []
    for i in ids:
        if i in (pad_idx, bos_idx):
            continue
        if i == eos_idx:
            break
        out.append(idx2char[i])
    return ""(out)