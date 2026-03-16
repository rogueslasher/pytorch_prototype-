
import argparse
import math
import os
import random
import string

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, global_step_from_engine
from ignite.metrics import Loss
from ignite.contrib.handlers import ProgressBar

CFG = {
    "min_len":        3,
    "max_len":        10,
    "train_size":     8000,
    "val_size":       1000,
    "batch_size":     128,
    "d_model":        128,
    "nhead":          4,
    "num_enc_layers": 2,
    "num_dec_layers": 2,
    "dim_feedforward": 256,
    "dropout":        0.1,
    "lr":             3e-4,
    "max_epochs":     30,
    "patience":       5,
    "checkpoint_dir": "./checkpoints/reverse",
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
}


CHARS = list(string.ascii_lowercase + string.digits)
PAD, BOS, EOS = "<PAD>", "<BOS>", "<EOS>"
VOCAB = [PAD, BOS, EOS] + CHARS

char2idx = {ch: i for i, ch in enumerate(VOCAB)}
idx2char = {i: ch for ch, i in char2idx.items()}

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

class ReverseStringDataset(Dataset):
    def __init__(self, size, min_len, max_len):
        self.data = []
        for _ in range(size):
            src = "".join(random.choices(CHARS, k=random.randint(min_len, max_len)))
            tgt = src[::-1]
            self.data.append((
                encode(src),
                [BOS_IDX] + encode(tgt),
                encode(tgt) + [EOS_IDX],
            ))

    def __len__(self):
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
    def infer(self, src_str, device, max_len=20):
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

def train():
    import os
    device = torch.device(CFG["device"])
    print(f"Device: {device}")

    train_ds = ReverseStringDataset(CFG["train_size"], CFG["min_len"], CFG["max_len"])
    val_ds   = ReverseStringDataset(CFG["val_size"],   CFG["min_len"], CFG["max_len"])
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
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
    PROBES = [("hello","olleh"), ("abcdef","fedcba"),
              ("python","nohtyp"), ("ignite","etingi")]

    @trainer.on(Events.EPOCH_COMPLETED(every=5))
    def qualitative_check(engine):
        print("\n── probes ───")
        for src_str, expected in PROBES:
            pred   = model.infer(src_str, device)
            status = "✓" if pred == expected else "✗"
            print(f"  {status}  '{src_str}' -> '{pred}'  (exp: '{expected}')")
        print()

    print(f"\nTraining for up to {CFG['max_epochs']} epochs...\n")
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
            pred = model.infer(args.infer, device)
            print(f"  Input   : '{args.infer}'")
            print(f"  Output  : '{pred}'")
            print(f"  Expected: '{args.infer[::-1]}'")
            print(f"  Correct : {'✓' if pred == args.infer[::-1] else '✗'}")
    else:
        train()