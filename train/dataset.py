import random
import torch
from datasets import load_dataset, IterableDataset

def text_stream(name, split, streaming):
    if name == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=streaming)
        for r in ds:
            yield r["text"]

    elif name == "math":
        ds = load_dataset("qwedsacf/competition_math", split=split, streaming=streaming)
        for r in ds:
            yield f"Problem:\n{r['problem']}\n\nSolution:\n{r['solution']}"

    elif name == "gsm8k":
        ds = load_dataset("openai/gsm8k", 'socratic', split=split, streaming=streaming)
        for r in ds:
            yield f"Problem:\n{r['question']}\n\nSolution:\n{r['answer']}"

    elif name == "dclm":
        ds = load_dataset("mlfoundations/dclm-pool-400m-1x", split=split, streaming=streaming)
        for r in ds:
            yield r["text"]

    elif name == "fineweb":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split=split, streaming=streaming, revision="main")
        for r in ds:
            if r.get("score", 0) >= 4:
                yield r["text"]


def chunk_stream(text_iter, tokenizer, cfg):
    step = cfg.chunk_size - cfg.chunk_overlap

    for item in text_iter:
        if isinstance(item, dict) and item.get("type") == "math":
            tokens = tokenizer.encode(item["text"], add_special_tokens=False)
            yield tokens[: cfg.chunk_size]
            continue

        tokens = tokenizer.encode(item, add_special_tokens=False)
        if len(tokens) < cfg.chunk_size:
            continue

        for i in range(0, len(tokens) - cfg.chunk_size + 1, step):
            yield tokens[i : i + cfg.chunk_size]


def build_llm_dataset(cfg, tokenizer, split="train", streaming=True, seed=42):
    random.seed(seed)

    sources = {
        "tinystories": cfg.percent_tinystories,
        "math":        cfg.percent_math,
        "gms8k":       cfg.percent_gms8k,
        "dclm":        cfg.percent_dclm,
        "fineweb":     cfg.percent_fineweb,
    }

    streams = {
        name: chunk_stream(
            text_stream(name, split, streaming),
            tokenizer,
            cfg,
        )
        for name, pct in sources.items()
        if pct > 0
    }

    names = list(streams.keys())
    weights = [sources[n] for n in names]

    total_tokens = cfg.total_tokens
    total_batches = total_tokens // (cfg.batch_size * cfg.chunk_size)

    causal_mask = torch.tril(
        torch.ones(cfg.chunk_size, cfg.chunk_size, dtype=torch.bool)
    ).unsqueeze(0).unsqueeze(0)

    def generator():
        produced_batches = 0

        while produced_batches < total_batches and names:
            batch = []

            while len(batch) < cfg.batch_size and names:
                packed = []

                while len(packed) < cfg.chunk_size and names:
                    name = random.choices(names, weights=weights, k=1)[0]
                    try:
                        chunk = next(streams[name])
                    except StopIteration:
                        # restart
                        chunk = chunk_stream(
                            text_stream(name, split, streaming),
                            tokenizer,
                            cfg,
                        )
                        continue

                    remaining = cfg.chunk_size - len(packed)
                    packed.extend(chunk[:remaining])

                if len(packed) == cfg.chunk_size:
                    batch.append(packed)

            if len(batch) < cfg.batch_size:
                break

            input_ids = torch.tensor(batch, dtype=torch.long)
            attention_mask = causal_mask.expand(cfg.batch_size, 1, cfg.chunk_size, cfg.chunk_size)

            produced_batches += 1
            dict_to_yield = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            yield dict_to_yield


    return IterableDataset.from_generator(generator)
