import random
import torch
from datasets import load_dataset, IterableDataset, disable_caching

disable_caching()


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
        ds = load_dataset("openai/gsm8k", "socratic", split=split, streaming=streaming)
        for r in ds:
            yield f"Problem:\n{r['question']}\n\nSolution:\n{r['answer']}"

    elif name == "dclm":
        ds = load_dataset("mlfoundations/dclm-pool-400m-1x", split=split, streaming=streaming)
        for r in ds:
            yield r["text"]

    elif name == "fineweb":
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split=split,
            streaming=streaming,
            revision="main",
        )
        for r in ds:
            if r.get("score", 0) >= 4:
                yield r["text"]


def chunk_stream(text_iter, tokenizer, cfg):
    step = cfg.chunk_size - cfg.chunk_overlap

    for text in text_iter:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < cfg.chunk_size:
            continue
        for i in range(0, len(tokens) - cfg.chunk_size + 1, step):
            yield tokens[i : i + cfg.chunk_size]


def build_llm_dataset(cfg, tokenizer, split="train", streaming=True, seed=42):
    sources = {
        "tinystories": cfg.percent_tinystories,
        "math": cfg.percent_math,
        "gsm8k": cfg.percent_gsm8k,
        "dclm": cfg.percent_dclm,
        "fineweb": cfg.percent_fineweb,
    }

    names = [k for k, v in sources.items() if v > 0]
    weights = [sources[n] for n in names]

    total_batches = cfg.total_tokens // (cfg.batch_size * cfg.chunk_size)

    causal_mask = torch.tril(
        torch.ones(cfg.chunk_size, cfg.chunk_size, dtype=torch.bool)
    ).unsqueeze(0).unsqueeze(0)

    def generator():
        rng = random.Random(seed)

        streams = {
            name: chunk_stream(
                text_stream(name, split, streaming),
                tokenizer,
                cfg,
            )
            for name in names
        }

        produced = 0

        while produced < total_batches and names:
            batch = []

            while len(batch) < cfg.batch_size:
                packed = []

                while len(packed) < cfg.chunk_size:
                    name = rng.choices(names, weights=weights, k=1)[0]

                    try:
                        chunk = next(streams[name])
                    except StopIteration:
                        streams[name] = chunk_stream(
                            text_stream(name, split, streaming),
                            tokenizer,
                            cfg,
                        )
                        continue

                    needed = cfg.chunk_size - len(packed)
                    packed.extend(chunk[:needed])

                batch.append(packed)

            input_ids = torch.tensor(batch, dtype=torch.long)
            attention_mask = causal_mask.expand(
                cfg.batch_size, 1, cfg.chunk_size, cfg.chunk_size
            )

            produced += 1
            yield {
                "input_ids": input_ids,
                "block_mask": attention_mask,
            }

    return IterableDataset.from_generator(generator)
