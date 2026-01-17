import math
from typing import Any, Dict, List, Optional

import torch
from torch.amp import GradScaler
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

from .config import TinyLogicLMConfig
from .model import load_model
from .dataset import build_llm_dataset

from torch.nn.attention import SDPBackend, sdpa_kernel

import wandb, os


def _create_scheduler(optimizer, num_steps: int, warmup_ratio: float = 0.02):
    warmup_steps = max(1, int(num_steps * warmup_ratio))
    min_lr_scale = 0.1

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_scale + (1 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda)


def _prepare_tokenizer():
    tok = AutoTokenizer.from_pretrained("./container/train/tokenizer/hf_tokenizer")
    tok.padding_side = "right"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def train(run: wandb.Run, cfg: TinyLogicLMConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = False # FP16 is too unstable for now (probably need to tune more hyperparameters)
    tokenizer = _prepare_tokenizer()

    model = load_model(
        cfg,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id or tokenizer.pad_token_id,
    ).to(device).to(torch.float16 if use_fp16 else torch.float32)

    dataloader = build_llm_dataset(cfg, tokenizer)

    tokens_per_step = cfg.batch_size * cfg.chunk_size * cfg.grad_accum_steps
    print(f"Tokens per step: {tokens_per_step}")
    total_steps = max(1, cfg.total_tokens // tokens_per_step)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr_max,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.adam_weight_decay,
    )
    scheduler = _create_scheduler(optimizer, total_steps, warmup_ratio=cfg.warmup_ratio)

    model.train()
    optimizer.zero_grad()
    if run is not None:
        run.watch(model, log_freq=100, log="all")

    global_step = 0
    micro_step = 0
    pbar = tqdm(total=total_steps, desc="train", leave=False)
    os.makedirs("./out", exist_ok=True)

    grad_norm = 0.0
    with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                                          SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]) if torch.cuda.is_available() else torch.enable_grad():
        data_iter = iter(dataloader)
        while global_step < total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["block_mask"],
                )
            logits = outputs.logits.float()  # [B, S, V]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            ) / cfg.grad_accum_steps

 
            loss.backward()
            micro_step += 1

            if global_step % 50 == 0:
                # Save model weights
                model_path = os.path.join("./out", f"microlm-{global_step}.pt")
                torch.save(model.state_dict(), model_path)
                if run is not None:
                    run.save(model_path, policy="now")



            if micro_step % cfg.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                true_loss = loss.item() * cfg.grad_accum_steps
                tokens_seen = global_step * tokens_per_step

                if run is not None:
                    run.log(
                        {
                            "train/loss": true_loss,
                            "train/lr": current_lr,
                            "train/grad_norm": grad_norm,
                            "train/global_step": global_step,
                            "train/tokens_seen": tokens_seen,
                        },
                        step=global_step,
                    )
                else:
                    # print(f"Step {global_step}: loss={true_loss:.4f}, lr={current_lr:.6e}, grad_norm={grad_norm:.4f}, tokens_seen={tokens_seen}")
                    pass
                pbar.set_postfix(loss=true_loss, lr=current_lr)
                pbar.update(1)

    pbar.close()


    # Save model weights
    model_path = os.path.join("./out", "microlm.pt")
    torch.save(model.state_dict(), model_path)
    if run is not None:
        run.save(model_path, policy="now")


    return model

if __name__ == "__main__":
    train(None, TinyLogicLMConfig())