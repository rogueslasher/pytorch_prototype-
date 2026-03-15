import argparse
import glob

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.models.transformer import Seq2SeqTransformer
from src.trainers.ignite_trainer import attach_handlers, build_evaluator, build_trainer
from ignite.engine import Events


def load_task(task, cfg):
    if task == "reverse":
        from src.datasets.string_reverse import (
            ReverseStringDataset, collate_fn,
            PAD_IDX, BOS_IDX, EOS_IDX, VOCAB_SIZE, char2idx, idx2char,
        )
        from src.vocab import encode, decode
        d = cfg["dataset"]
        train_ds = ReverseStringDataset(d["train_size"], d["min_len"], d["max_len"])
        val_ds   = ReverseStringDataset(d["val_size"],   d["min_len"], d["max_len"])
        enc = lambda s: encode(s, char2idx)
        dec = lambda ids: decode(ids, idx2char, PAD_IDX, BOS_IDX, EOS_IDX)
        vocab = dict(vocab_size=VOCAB_SIZE, pad_idx=PAD_IDX,
                     bos_idx=BOS_IDX, eos_idx=EOS_IDX, encode=enc, decode=dec)
        probes = [("hello","olleh"),("abcdef","fedcba"),
                  ("python","nohtyp"),("ignite","etingi")]
        return train_ds, val_ds, collate_fn, vocab, probes, None

    elif task == "addition":
        from src.datasets.integer_addition import (
            AdditionDataset, collate_fn,
            PAD_IDX, BOS_IDX, EOS_IDX, VOCAB_SIZE, char2idx, idx2char,
            _random_sample,
        )
        from src.vocab import encode, decode
        d = cfg["dataset"]
        train_ds = AdditionDataset(d["train_size"], d["train_digits"])
        val_ds   = AdditionDataset(d["val_size"],   d["train_digits"])
        enc = lambda s: encode(s, char2idx)
        dec = lambda ids: decode(ids, idx2char, PAD_IDX, BOS_IDX, EOS_IDX)
        vocab = dict(vocab_size=VOCAB_SIZE, pad_idx=PAD_IDX,
                     bos_idx=BOS_IDX, eos_idx=EOS_IDX, encode=enc, decode=dec)
        probes = [("12+34", str(12+34)), ("999+1", str(999+1)),
                  ("9999+9999", str(9999+9999)), ("12345+67890", str(12345+67890))]
        gen_info = (_random_sample, d["train_digits"], d["gen_digits"])
        return train_ds, val_ds, collate_fn, vocab, probes, gen_info

    else:
        raise ValueError(f"Unknown task: {task!r}")


def run_generalization(model, vocab, gen_info, device):
    random_sample_fn, train_digits, gen_digits = gen_info
    model.eval()
    results = {}
    for _ in range(1000):
        src_str, expected = random_sample_fn(gen_digits)
        pred_ids  = model.infer(src_str, vocab["encode"], vocab["bos_idx"],
                                vocab["eos_idx"], vocab["pad_idx"], device)
        predicted = vocab["decode"](pred_ids)
        a, b = src_str.split("+")
        d = max(len(a), len(b))
        if d not in results:
            results[d] = [0, 0]
        results[d][1] += 1
        if predicted == expected:
            results[d][0] += 1

    print("\n── Generalization Report ─────────────────────────────────")
    print(f"  {'Digits':>6} | {'Correct':>7} | {'Total':>5} | {'Acc':>7} | Status")
    print(f"  {'─'*6}-+-{'─'*7}-+-{'─'*5}-+-{'─'*7}-+--------")
    in_c = in_t = ood_c = ood_t = 0
    for d in sorted(results):
        c, t  = results[d]
        acc   = c / t * 100
        label = "in-dist" if d <= train_digits else "OOD ⚡"
        print(f"  {d:>6} | {c:>7} | {t:>5} | {acc:>6.1f}% | {label}")
        if d <= train_digits:
            in_c += c; in_t += t
        else:
            ood_c += c; ood_t += t
    print()
    if in_t:  print(f"  In-distribution : {in_c/in_t*100:.1f}%")
    if ood_t: print(f"  OOD             : {ood_c/ood_t*100:.1f}%")
    print("──────────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   default="addition", choices=["reverse","addition"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--infer",  default=None)
    args = parser.parse_args()

    config_path = args.config or f"configs/{args.task}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    task = cfg.get("task", args.task)
    t, m = cfg["training"], cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Task: {task}  |  Device: {device}")

    train_ds, val_ds, collate_fn, vocab, probes, gen_info = load_task(task, cfg)

    train_loader = DataLoader(train_ds, batch_size=t["batch_size"],
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=t["batch_size"],
                              shuffle=False, collate_fn=collate_fn)
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    model = Seq2SeqTransformer(
        vocab_size=vocab["vocab_size"], d_model=m["d_model"], nhead=m["nhead"],
        num_encoder_layers=m["num_encoder_layers"],
        num_decoder_layers=m["num_decoder_layers"],
        dim_feedforward=m["dim_feedforward"], dropout=m["dropout"],
        pad_idx=vocab["pad_idx"],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    if args.infer:
        ckpts = sorted(glob.glob(f"{t['checkpoint_dir']}/checkpoint_*.pt"))
        if not ckpts:
            print("No checkpoint found — train first.")
            return
        ckpt = torch.load(ckpts[-1], map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        pred = vocab["decode"](model.infer(
            args.infer, vocab["encode"],
            vocab["bos_idx"], vocab["eos_idx"], vocab["pad_idx"], device
        ))
        print(f"  Input : '{args.infer}'\n  Output: '{pred}'")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["pad_idx"])
    trainer   = build_trainer(model, optimizer, criterion, device, vocab["pad_idx"])
    evaluator = build_evaluator(model, criterion, device, vocab["pad_idx"])
    attach_handlers(trainer, evaluator, model, optimizer,
                    val_loader, t["checkpoint_dir"], patience=t["patience"])

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def qualitative_check(engine):
        print("\n── probes ───")
        for src_str, expected in probes:
            pred   = vocab["decode"](model.infer(
                src_str, vocab["encode"],
                vocab["bos_idx"], vocab["eos_idx"], vocab["pad_idx"], device
            ))
            status = "✓" if pred == expected else "✗"
            print(f"  {status}  {src_str} -> {pred}  (exp: {expected})")
        print()

    if gen_info:
        @trainer.on(Events.COMPLETED)
        def final_gen_report(engine):
            run_generalization(model, vocab, gen_info, device)

    print(f"Training for up to {t['max_epochs']} epochs...\n")
    trainer.run(train_loader, max_epochs=t["max_epochs"])
    print("Done.")


if __name__ == "__main__":
    main()
