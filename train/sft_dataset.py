import json
import math
import os

import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


def build_chat_llm_dataset(
    cfg,
    tokenizer,
    fallback_tokenizer=None,
    *,
    local_json_path: str,
):
    """
    Local JSON conversations → tokenized DatasetDict with seg_ids → HF upload.

    Splits:
      - train: 92.5%
      - validation: 5%
      - test: 2.5%

    Cached fast-path:
      If cfg.hf_dataset_name exists on HF, load and return it.
    """

    # ------------------------------------------------------------
    # Fast path: already uploaded dataset
    # ------------------------------------------------------------
    if cfg.hf_dataset_name is not None:
        try:
            print(f"Loading cached dataset from HF: {cfg.hf_dataset_name}")
            return load_dataset(cfg.hf_dataset_name, split="train")
        except Exception as e:
            print(e)
            print("Cached dataset not found, rebuilding...")

    # ------------------------------------------------------------
    # Patch tokenizer for chat tokens
    # ------------------------------------------------------------
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
    )
    if fallback_tokenizer is not None:
        fallback_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
        )
    eos = tokenizer.eos_token_id

    chunk_size = cfg.chunk_size
    batch_size = cfg.batch_size
    total_batches = cfg.total_tokens // (batch_size * chunk_size)

    # ------------------------------------------------------------
    # Load local JSON (pure Python)
    # ------------------------------------------------------------
    print(f"Loading conversations from {local_json_path}")
    with open(local_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ------------------------------------------------------------
    # Tokenization helper
    # ------------------------------------------------------------
    def tokenize_conversation(conv):
        tokens = []
        segs = []
        seg_id = 0

        for msg in conv:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if not role or not content:
                continue

            text = f"<|im_start|>{role}\n{content}<|im_end|>\n"

            enc = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]

            if (
                tokenizer.unk_token_id is not None
                and tokenizer.unk_token_id in enc
                and fallback_tokenizer is not None
            ):
                enc = fallback_tokenizer(
                    text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]

            tokens.extend(enc)
            segs.extend([seg_id] * len(enc))
            seg_id += 1

        tokens.append(eos)
        segs.append(seg_id)
        return tokens, segs

    # ------------------------------------------------------------
    # Pack into fixed-size batches (with tqdm)
    # ------------------------------------------------------------
    all_input_ids = []
    all_seg_ids = []

    buffer_tokens = []
    buffer_segs = []
    produced = 0

    batch_pbar = tqdm(
        total=total_batches,
        desc="Packing token batches",
        unit="batch",
    )

    for row in tqdm(data["conversations"], desc="Reading conversations", unit="conv"):
        if produced >= total_batches:
            break

        try:
            toks, segs = tokenize_conversation(row)
            if not toks:
                continue
        except Exception as e:
            print(f"Skipping conversation due to error: {e}")
            continue

        buffer_tokens.extend(toks)
        buffer_segs.extend(segs)

        while len(buffer_tokens) >= batch_size * chunk_size and produced < total_batches:
            batch_tokens = []
            batch_segs = []

            for _ in range(batch_size):
                batch_tokens.append(buffer_tokens[:chunk_size])
                batch_segs.append(buffer_segs[:chunk_size])

                buffer_tokens = buffer_tokens[chunk_size:]
                buffer_segs = buffer_segs[chunk_size:]

            all_input_ids.append(batch_tokens)
            all_seg_ids.append(batch_segs)

            produced += 1
            batch_pbar.update(1)

    batch_pbar.close()

    # ------------------------------------------------------------
    # Create Dataset
    # ------------------------------------------------------------
    print("Creating HuggingFace Dataset object...")
    ds = Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "seg_ids": all_seg_ids,
        }
    )

    # ------------------------------------------------------------
    # Split: 92.5 / 5 / 2.5
    # ------------------------------------------------------------
    n = len(ds)
    n_test = math.floor(0.025 * n)
    n_val = math.floor(0.05 * n)
    n_train = n - n_val - n_test

    ds = ds.shuffle(seed=42)

    dataset_dict = DatasetDict(
        {
            "train": ds.select(range(0, n_train)),
            "validation": ds.select(range(n_train, n_train + n_val)),
            "test": ds.select(range(n_train + n_val, n)),
        }
    )

    print(
        f"Dataset split sizes → "
        f"train={n_train}, val={n_val}, test={n_test}"
    )

    # ------------------------------------------------------------
    # Upload to HF
    # ------------------------------------------------------------
    if cfg.hf_dataset_name is not None:
        if cfg.hf_token is not None:
            os.environ["HF_TOKEN"] = cfg.hf_token

        print(f"Uploading dataset to HF repo: {cfg.hf_dataset_name}")
        dataset_dict.push_to_hub(cfg.hf_dataset_name)

    return dataset_dict


def main():
    from .config import TinyLogicLMConfig
    import argparse, json
    from transformers import AutoTokenizer
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    args = p.parse_args()

    with open(args.cfg) as f:
        cfg = TinyLogicLMConfig(**(json.load(f)["input"]["config"]))

    file_path = Path(__file__).resolve()
    tokenizer_path = file_path.parent / "tokenizer" / "hf_tokenizer"
    tokenizerSlow = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    tokenizerFast = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    _ = build_chat_llm_dataset(
        cfg,
        tokenizerFast,
        tokenizerSlow,
        local_json_path="./train/fireworks_ingested_conversations.json",
    )


if __name__ == "__main__":
    main()
