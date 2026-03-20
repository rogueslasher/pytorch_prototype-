
Two tasks are implemented as end-to-end training experiments:

| Task | Input | Output |
|------|-------|--------|
| String reversal | `"abcdef"` | `"fedcba"` |
| Integer addition | `"123+456"` | `"579"` |

---

## Structure
```
prototype/
├── main.py                        
├── requirements.txt
├── README.md                      
├── .gitignore
├── configs/
│   ├── reverse.yaml
│   └── addition.yaml
├── src/
│   ├── __init__.py
│   ├── vocab.py                   
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── string_reverse.py
│   │   └── integer_addition.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer.py
│   └── trainers/
│       ├── __init__.py
│       └── ignite_trainer.py
└── scripts/
    ├── train_reverse.py
    └── train_addition.py
```

---

## Quickstart
```bash
pip install -r requirements.txt

# train string reversal
python scripts/train_reverse.py

# train integer addition (includes generalization report)
python scripts/train_addition.py

# inference
python scripts/train_reverse.py --infer "helloworld"
python scripts/train_addition.py --infer "999+1"

# use main.py with YAML config
python main.py --task reverse
python main.py --task addition
```

---

## PyTorch-Ignite usage

| Ignite component | Where used |
|-----------------|------------|
| `Engine` | `build_trainer`, `build_evaluator` |
| `Events.EPOCH_COMPLETED` | validation loop, LR scheduling, qualitative probes |
| `Events.COMPLETED` | generalization report |
| `Loss` metric | attached to evaluator, aggregates over full val set |
| `EarlyStopping` | stops training if val_loss stalls |
| `Checkpoint` + `DiskSaver` | saves best model automatically |
| `ProgressBar` | per-iteration loss display |

---

## Results

### String Reversal
- val_loss: `0.0054` after 30 epochs on CPU
- All probes correct by epoch 10

| Probe | Expected | Result |
|-------|----------|--------|
| hello | olleh | ✓ |
| abcdef | fedcba | ✓ |
| python | nohtyp | ✓ |
| ignite | etingi | ✓ |

### Integer Addition — Generalization Experiment

Trained on additions with up to **3 digits** per operand.
Tested on up to **6 digits** (OOD = beyond training distribution).

| Digits | Correct | Total | Accuracy | Status |
|--------|---------|-------|----------|--------|
| 1 | 4 | 197 | 2.0% | in-dist |
| 2 | 137 | 150 | 91.3% | in-dist |
| 3 | 177 | 178 | 99.4% | in-dist |
| 4 | 0 | 161 | 0.0% | OOD ⚡ |
| 5 | 0 | 153 | 0.0% | OOD ⚡ |
| 6 | 0 | 161 | 0.0% | OOD ⚡ |

**In-distribution: 60.6% — OOD: 0.0%**

The sharp accuracy drop beyond the training distribution demonstrates the
**length generalization problem** — the key benchmark `trainite` should make
easy to reproduce across different architectures (Transformer vs Mamba vs RWKV).

---

## Connection to trainite

This prototype is a direct proof-of-concept for the GSoC 2026 project.

The `src/` structure here — separate `datasets/`, `models/`, `trainers/` with
a thin `main.py` wiring them via YAML config — is the skeleton that `trainite`
would make installable and configurable.

The generalization experiment above is exactly the kind of benchmark `trainite`
should automate: swap `Seq2SeqTransformer` for Mamba or RWKV via config, run
the same experiment, compare who handles length generalization better.