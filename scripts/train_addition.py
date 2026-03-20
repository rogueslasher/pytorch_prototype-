import argparse
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, global_step_from_engine
from ignite.metrics import Loss
from ignite.contrib.handlers import ProgressBar

CFG = {
    "train_digits":   3,      # was 5 — start simpler, up to 999+999
    "gen_digits":     6,      # was 10
    "train_size":     20000,
    "val_size":       2000,
    "gen_size":       500,
    "batch_size":     256,
    "d_model":        128,
    "nhead":          4,
    "num_enc_layers": 3,
    "num_dec_layers": 3,
    "dim_feedforward": 512,   
    "dropout":        0.0,    
    "lr":             1e-3,   
    "max_epochs":     40,
    "patience":       10,
    "checkpoint_dir": "./checkpoints/addition",
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
}

CHARS = list("0123456789+")
PAD, BOS, EOS = "<PAD>", "<BOS>", "<EOS>"
VOCAB = [PAD, BOS, EOS] + CHARS

char2idx  = {ch: i for i, ch in enumerate(VOCAB)}
idx2char  = {i: ch for ch, i in char2idx.items()}

PAD_IDX   = char2idx[PAD]
BOS_IDX   = char2idx[BOS]
EOS_IDX   = char2idx[EOS]
VOCAB_SIZE = len(VOCAB)


def encode(s):
    return [char2idx[c] for c in s]


def decode(ids):
    out = []
    for i in ids:
        ch = idx2char[i]
        if ch in (PAD, BOS):
            continue
        if ch == EOS:
            break
        out.append(ch)
    return "".join(out)


def random_sample(max_digits):
    max_val = 10 ** max_digits - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    return f"{a}+{b}", str(a + b)


class AdditionDataset(Dataset):
    def __init__(self, size, max_digits):
        self.data = []
        seen = set()
        while len(self.data) < size:
            src, tgt = random_sample(max_digits)
            if src in seen:
                continue
            seen.add(src)
            self.data.append((
                encode(src),
                [BOS_IDX] + encode(tgt),
                encode(tgt) + [EOS_IDX],
                src, tgt,
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_in, tgt_out, _, _ = self.data[idx]
        return src_ids, tgt_in, tgt_out

    def raw(self, idx):
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead,
                 num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, pad_idx):
        super().__init__()
        self.d_model   = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        T        = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=src.device)
        src_emb  = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb  = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer(
            src_emb, tgt_emb, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(out)

    @torch.no_grad()
    def infer(self, src_str, device, max_len=30):
        self.eval()
        src_ids = torch.tensor([encode(src_str)], dtype=torch.long, device=device)
        src_pad = (src_ids == PAD_IDX)
        src_emb = self.pos_enc(self.src_embed(src_ids) * math.sqrt(self.d_model))
        memory  = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad)

        tgt_ids = [BOS_IDX]
        for _ in range(max_len):
            tgt_t    = torch.tensor([tgt_ids], dtype=torch.long, device=device)
            tgt_emb  = self.pos_enc(self.tgt_embed(tgt_t) * math.sqrt(self.d_model))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_t.size(1), device=device
            )
            out     = self.transformer.decoder(
                tgt_emb, memory, tgt_mask=tgt_mask,
                memory_key_padding_mask=src_pad
            )
            next_id = self.fc_out(out)[0, -1].argmax(-1).item()
            if next_id == EOS_IDX:
                break
            tgt_ids.append(next_id)
        return decode(tgt_ids)

def build_trainer(model, optimizer, criterion, device):
    def train_step(engine, batch):
        model.train()
        src, tgt_in, tgt_out = [t.to(device) for t in batch]
        src_pad = (src == PAD_IDX)
        tgt_pad = (tgt_in == PAD_IDX)
        logits  = model(src, tgt_in,
                        src_key_padding_mask=src_pad,
                        tgt_key_padding_mask=tgt_pad)
        B, T, V = logits.shape
        loss    = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()
    return Engine(train_step)


def build_evaluator(model, criterion, device):
    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            src, tgt_in, tgt_out = [t.to(device) for t in batch]
            src_pad = (src == PAD_IDX)
            tgt_pad = (tgt_in == PAD_IDX)
            logits  = model(src, tgt_in,
                            src_key_padding_mask=src_pad,
                            tgt_key_padding_mask=tgt_pad)
            B, T, V = logits.shape
            return logits.reshape(B * T, V), tgt_out.reshape(B * T)
    evaluator = Engine(eval_step)
    Loss(nn.CrossEntropyLoss(ignore_index=PAD_IDX)).attach(evaluator, "val_loss")
    return evaluator

def run_generalization(model, device):
    model.eval()
    results = {}
    for _ in range(1000):
        d = random.randint(1, CFG["gen_digits"])
        max_val = 10 ** d - 1
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        src_str, expected = f"{a}+{b}", str(a + b)
        predicted = model.infer(src_str, device)
        correct   = (predicted == expected)
        a, b = src_str.split("+")
        d = max(len(a), len(b))
        if d not in results:
            results[d] = [0, 0]
        results[d][1] += 1
        if correct:
            results[d][0] += 1

    print("\n── Generalization Report ─────────────────────────────────")
    print(f"  {'Digits':>6} | {'Correct':>7} | {'Total':>5} | {'Acc':>7} | Status")
    print(f"  {'─'*6}-+-{'─'*7}-+-{'─'*5}-+-{'─'*7}-+--------")
    in_c = in_t = ood_c = ood_t = 0
    for d in sorted(results):
        c, t  = results[d]
        acc   = c / t * 100
        label = "in-dist" if d <= CFG["train_digits"] else "OOD ⚡"
        print(f"  {d:>6} | {c:>7} | {t:>5} | {acc:>6.1f}% | {label}")
        if d <= CFG["train_digits"]:
            in_c += c; in_t += t
        else:
            ood_c += c; ood_t += t
    print()
    if in_t:  print(f"  In-distribution : {in_c/in_t*100:.1f}%  ({in_c}/{in_t})")
    if ood_t: print(f"  OOD             : {ood_c/ood_t*100:.1f}%  ({ood_c}/{ood_t})")
    print("──────────────────────────────────────────────────────────\n")

def train():
    import os
    device = torch.device(CFG["device"])
    print(f"Device: {device}")
    print(f"Train on ≤{CFG['train_digits']} digits, test generalization on ≤{CFG['gen_digits']} digits\n")

    print("Generating datasets...")
    train_ds = AdditionDataset(CFG["train_size"], CFG["train_digits"])
    val_ds   = AdditionDataset(CFG["val_size"],   CFG["train_digits"])
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, collate_fn=collate_fn)

    model = Seq2SeqTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=CFG["d_model"], nhead=CFG["nhead"],
        num_encoder_layers=CFG["num_enc_layers"],
        num_decoder_layers=CFG["num_dec_layers"],
        dim_feedforward=CFG["dim_feedforward"],
        dropout=CFG["dropout"], pad_idx=PAD_IDX,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    trainer   = build_trainer(model, optimizer, criterion, device)
    evaluator = build_evaluator(model, criterion, device)

    ProgressBar(persist=False).attach(trainer, output_transform=lambda x: {"loss": x})

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        evaluator.run(val_loader)
        val_loss = evaluator.state.metrics["val_loss"]
        scheduler.step(val_loss)
        print(
            f"Epoch {engine.state.epoch:3d} | "
            f"train_loss={engine.state.output:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    def score_fn(engine):
        return -engine.state.metrics["val_loss"]

    EarlyStopping(patience=CFG["patience"], score_function=score_fn,
                  trainer=trainer).attach(evaluator)

    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    checkpoint_handler = Checkpoint(
        to_save={"model": model, "optimizer": optimizer},
        save_handler=DiskSaver(CFG["checkpoint_dir"], create_dir=True, require_empty=False),
        n_saved=1, score_function=score_fn, score_name="neg_val_loss",
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)

    PROBES = [
        ("12+34",       str(12 + 34)),
        ("999+1",       str(999 + 1)),
        ("9999+9999",   str(9999 + 9999)),
        ("12345+67890", str(12345 + 67890)),
    ]

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def qualitative_check(engine):
        print("\n── probes ───")
        for src_str, expected in PROBES:
            pred   = model.infer(src_str, device)
            status = "✓" if pred == expected else "✗"
            print(f"  {status}  {src_str} = {pred}  (exp: {expected})")
        print()

    @trainer.on(Events.COMPLETED)
    def final_gen_report(engine):
        run_generalization(model, device)

    print(f"Training for up to {CFG['max_epochs']} epochs...\n")
    trainer.run(train_loader, max_epochs=CFG["max_epochs"])
    print("Done.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str, default=None)
    args = parser.parse_args()

    if args.infer:
        import glob
        device = torch.device(CFG["device"])
        model  = Seq2SeqTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=CFG["d_model"], nhead=CFG["nhead"],
            num_encoder_layers=CFG["num_enc_layers"],
            num_decoder_layers=CFG["num_dec_layers"],
            dim_feedforward=CFG["dim_feedforward"],
            dropout=CFG["dropout"], pad_idx=PAD_IDX,
        ).to(device)
        ckpts = sorted(glob.glob(f"{CFG['checkpoint_dir']}/checkpoint_*.pt"))
        if not ckpts:
            print("No checkpoint found. Train first.")
        else:
            ckpt = torch.load(ckpts[-1], map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            a, b     = args.infer.split("+")
            expected = str(int(a) + int(b))
            pred     = model.infer(args.infer, device)
            print(f"  Input   : '{args.infer}'")
            print(f"  Output  : '{pred}'")
            print(f"  Expected: '{expected}'")
            print(f"  Correct : {'✓' if pred == expected else '✗'}")
    else:
        train()