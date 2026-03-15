import os
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import (
    Checkpoint, DiskSaver, EarlyStopping, global_step_from_engine,
)
from ignite.metrics import Loss


def _padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return seq == pad_idx


def build_trainer(model, optimizer, criterion, device, pad_idx) -> Engine:
    def train_step(engine, batch):
        model.train()
        src, tgt_in, tgt_out = [t.to(device) for t in batch]
        logits  = model(src, tgt_in,
                        src_key_padding_mask=_padding_mask(src, pad_idx),
                        tgt_key_padding_mask=_padding_mask(tgt_in, pad_idx))
        B, T, V = logits.shape
        loss    = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()
    return Engine(train_step)


def build_evaluator(model, criterion, device, pad_idx) -> Engine:
    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            src, tgt_in, tgt_out = [t.to(device) for t in batch]
            logits  = model(src, tgt_in,
                            src_key_padding_mask=_padding_mask(src, pad_idx),
                            tgt_key_padding_mask=_padding_mask(tgt_in, pad_idx))
            B, T, V = logits.shape
            return logits.reshape(B * T, V), tgt_out.reshape(B * T)
    evaluator = Engine(eval_step)
    Loss(nn.CrossEntropyLoss(ignore_index=pad_idx)).attach(evaluator, "val_loss")
    return evaluator


def attach_handlers(trainer, evaluator, model, optimizer,
                    val_loader, checkpoint_dir, patience=7):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
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

    EarlyStopping(patience=patience, score_function=score_fn,
                  trainer=trainer).attach(evaluator)

    os.makedirs(checkpoint_dir, exist_ok=True)
    Checkpoint(
        to_save={"model": model, "optimizer": optimizer},
        save_handler=DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
        n_saved=1,
        score_function=score_fn,
        score_name="neg_val_loss",
        global_step_transform=global_step_from_engine(trainer),
    ).attach(evaluator)
