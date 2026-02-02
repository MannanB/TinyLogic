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
from .sft_dataset import build_chat_llm_dataset
from .dataset import _build_block_diag_causal_mask
from .train import _create_scheduler, _prepare_tokenizer

from torch.nn.attention import SDPBackend, sdpa_kernel

import wandb, os
from pathlib import Path



def patch_vocab(model: torch.nn.Module, tokenizer: AutoTokenizer):
    """
    Adds chat special tokens to tokenizer and resizes model embeddings + lm_head.

    Assumes:
    - model.model.embed_tokens is Gemma3TextScaledWordEmbedding
    - model.lm_head is nn.Linear (no bias)
    """

    special_tokens = ["<|im_start|>", "<|im_end|>"]

    # 1. Add tokens to tokenizer
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )
    if num_added == 0:
        return  # already patched

    # 2. Grab old modules
    embed = model.model.embed_tokens
    lm_head = model.lm_head

    old_vocab_size, hidden_size = embed.weight.shape
    new_vocab_size = old_vocab_size + num_added
    device = embed.weight.device
    dtype = embed.weight.dtype

    # 3. Create new embedding (preserve embed_scale!)
    new_embed = embed.__class__(
        num_embeddings=new_vocab_size,
        embedding_dim=hidden_size,
        padding_idx=embed.padding_idx,
        embed_scale=float(embed.embed_scale),
    ).to(device=device, dtype=dtype)

    # copy old weights
    new_embed.weight.data[:old_vocab_size] = embed.weight.data

    # init new token embeddings
    torch.nn.init.normal_(
        new_embed.weight.data[old_vocab_size:],
        mean=0.0,
        std=hidden_size ** -0.5,
    )

    # 4. Replace embedding
    model.model.embed_tokens = new_embed

    # 5. Resize lm_head
    new_lm_head = torch.nn.Linear(
        hidden_size,
        new_vocab_size,
        bias=False,
        device=device,
        dtype=dtype,
    )

    new_lm_head.weight.data[:old_vocab_size] = lm_head.weight.data
    torch.nn.init.normal_(
        new_lm_head.weight.data[old_vocab_size:],
        mean=0.0,
        std=hidden_size ** -0.5,
    )

    model.lm_head = new_lm_head

    # 6. Re-tie weights if originally tied
    if embed.weight.data_ptr() == lm_head.weight.data_ptr():
        model.lm_head.weight = model.model.embed_tokens.weight


def train(run: wandb.Run, load_path: str, cfg: TinyLogicLMConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = False # FP16 is too unstable for now (probably need to tune more hyperparameters)
    tokenizer = _prepare_tokenizer()

    model = load_model(
        cfg,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id or tokenizer.pad_token_id,
        load_from_hf=not (load_path is None),
        hf_load_path="MannanB/tinylogic-base",
    ).to(device).to(torch.float16 if use_fp16 else torch.float32)

    patch_vocab(model, tokenizer)

    dataloader = build_chat_llm_dataset(cfg, tokenizer, local_json_path=None)

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

    causal = torch.ones(cfg.chunk_size, cfg.chunk_size, dtype=torch.bool).tril_().to(device)

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

            dtype_map = {"input_ids": torch.long, "seg_ids": torch.int32}
            batch = {k: torch.tensor(v, dtype=dtype_map[k]).to(device) for k, v in batch.items()}

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=_build_block_diag_causal_mask(batch["seg_ids"], causal),
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

            if global_step % 500 == 0:
                # Save model weights
                model_path = os.path.join("./out", f"microlm-sft-{global_step}.pt")
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
    model_path = os.path.join("./out", "microlm-sft.pt")
    torch.save(model.state_dict(), model_path)
    if run is not None:
        run.save(model_path, policy="now")


    return model

if __name__ == "__main__":
    from .config import TinyLogicLMConfig
    import argparse, json
    from transformers import AutoTokenizer
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    args = p.parse_args()

    with open(args.cfg) as f:
        cfg = TinyLogicLMConfig(**(json.load(f)["input"]["config"]))



    train(None, None, cfg)