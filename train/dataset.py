import random
import itertools
from collections import deque

import torch
from datasets import load_dataset, IterableDataset, disable_caching, Dataset
import os
from tqdm import tqdm
import tempfile

disable_caching()


# --- (kept for compatibility; not used internally anymore, but fine to keep) ---
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


# --- fast, restartable, batched-tokenization source stream ---
class _TokenSource:
    """
    Produces token chunks from a dataset, but *packs many short docs* into fixed-length sequences.
    Maintains internal state so long docs can continue across sequences with overlap.
    """

    def __init__(
        self,
        name: str,
        split: str,
        streaming: bool,
        tokenizer,
        fallback_tokenizer,
        seed: int,
        *,
        shuffle_buffer: int,
        batch_texts: int,
        max_doc_tokens: int,
        min_doc_tokens: int,
        chunk_overlap: int,
    ):
        self.name = name
        self.split = split
        self.streaming = streaming
        self.tokenizer = tokenizer
        self.fallback_tokenizer = fallback_tokenizer
        self.seed = seed

        self.shuffle_buffer = shuffle_buffer
        self.batch_texts = batch_texts
        self.max_doc_tokens = max_doc_tokens
        self.min_doc_tokens = min_doc_tokens
        self.chunk_overlap = chunk_overlap

        self._pending_docs = deque()
        self._cur_tokens = None
        self._cur_pos = 0

        self._reset_iter()

    def _reset_iter(self):
        # Load + shuffle at the dataset level (way faster/better than python-level shuffling).
        if self.name == "tinystories":
            ds = load_dataset("roneneldan/TinyStories", split=self.split, streaming=self.streaming)
            get_text = lambda r: r["text"]

        elif self.name == "math":
            ds = load_dataset("qwedsacf/competition_math", split=self.split, streaming=self.streaming)
            get_text = lambda r: f"Problem:\n{r['problem']}\n\nSolution:\n{r['solution']}"

        elif self.name == "gsm8k":
            ds = load_dataset("openai/gsm8k", "socratic", split=self.split, streaming=self.streaming)
            get_text = lambda r: f"Problem:\n{r['question']}\n\nSolution:\n{r['answer']}"

        elif self.name == "dclm":
            ds = load_dataset("mlfoundations/dclm-pool-400m-1x", split=self.split, streaming=self.streaming)
            get_text = lambda r: r["text"]

        elif self.name == "fineweb":
            ds = load_dataset("HuggingFaceFW/fineweb-edu", split=self.split, streaming=self.streaming, revision="main")
            # filter later, because streaming shuffle happens at ds-level
            def get_text(r):
                return r["text"] if r.get("score", 0) >= 4 else None

        else:
            raise ValueError(f"Unknown source: {self.name}")

        if self.streaming:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        else:
            ds = ds.shuffle(seed=self.seed)

        self._row_iter = iter(ds)
        self._get_text = get_text

    def _refill_docs(self):
        eos = self.tokenizer.eos_token_id
        # Pull a batch of texts; restart if exhausted.
        texts = []
        while len(texts) < self.batch_texts:
            try:
                r = next(self._row_iter)
            except StopIteration:
                self._reset_iter()
                continue

            t = self._get_text(r)
            if not t:
                continue
            texts.append(t)

        enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        if (self.tokenizer.unk_token_id in enc):
            print("Falling back")
            enc = self.fallback_tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        for ids in enc["input_ids"]:
            if len(ids) < self.min_doc_tokens:
                continue
            if len(ids) > self.max_doc_tokens:
                ids = ids[: self.max_doc_tokens]
            ids.append(eos)
            self._pending_docs.append(ids)

    def _ensure_current_doc(self):
        while self._cur_tokens is None or self._cur_pos >= len(self._cur_tokens):
            if not self._pending_docs:
                self._refill_docs()
            self._cur_tokens = self._pending_docs.popleft()
            self._cur_pos = 0

    def take_up_to(self, max_len: int):
        """
        Take up to max_len tokens from the current doc.
        Returns (piece, ended_doc: bool).
        """
        self._ensure_current_doc()
        n = min(max_len, len(self._cur_tokens) - self._cur_pos)
        piece = self._cur_tokens[self._cur_pos : self._cur_pos + n]
        self._cur_pos += n
        ended = self._cur_pos >= len(self._cur_tokens)
        return piece, ended

    def rewind_overlap_if_mid_doc(self):
        # If we stopped mid-doc due to sequence boundary, re-include a suffix as context in the next chunk.
        if self._cur_tokens is None:
            return
        if 0 < self._cur_pos < len(self._cur_tokens) and self.chunk_overlap > 0:
            self._cur_pos = max(0, self._cur_pos - self.chunk_overlap)


def _build_block_diag_causal_mask(seg_ids: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
    """
    seg_ids: (B, S) int
    causal:  (S, S) bool with tril(True)
    Returns: (B, 1, S, S) bool where attend is allowed iff:
      - causal, and
      - same segment id, and
      - segment id != -1 (valid)
    """
    # (B, S)
    valid = seg_ids.ne(-1)
    # (B, S, S)
    same_seg = seg_ids.unsqueeze(2).eq(seg_ids.unsqueeze(1))
    m = same_seg & causal.unsqueeze(0)
    m = m & valid.unsqueeze(2) & valid.unsqueeze(1)
    return m.unsqueeze(1)


def build_llm_dataset(cfg, tokenizer, fallback_tokenizer=None, split="train", streaming=True, seed=42):
    # ---- fast path: load cached HF dataset if provided ----
    if cfg.hf_dataset_name is not None:
        try:
            return load_dataset(cfg.hf_dataset_name, split=split)
        except Exception as e:
            print(e)
            pass  # fall through and rebuild

    sources = {
        "tinystories": cfg.percent_tinystories,
        "math": cfg.percent_math,
        "gsm8k": cfg.percent_gsm8k,
        "dclm": cfg.percent_dclm,
        "fineweb": cfg.percent_fineweb,
    }

    names = [k for k, v in sources.items() if v > 0]
    weights = [sources[n] for n in names]
    if not names:
        raise ValueError("No dataset sources enabled.")

    total_batches = cfg.total_tokens // (cfg.batch_size * cfg.chunk_size)
    causal = torch.ones(cfg.chunk_size, cfg.chunk_size, dtype=torch.bool).tril_()

    shuffle_buffer = getattr(cfg, "shuffle_buffer", 1_000)
    batch_texts = getattr(cfg, "tokenize_batch_texts", 256)
    min_doc_tokens = getattr(cfg, "min_doc_tokens", 1)
    max_doc_tokens = getattr(cfg, "max_doc_tokens", cfg.chunk_size * 8)

    print("total batches:", total_batches)

    def generator():
        rng = random.Random(seed)

        streams = {
            name: _TokenSource(
                name=name,
                split=split,
                streaming=streaming,
                tokenizer=tokenizer,
                fallback_tokenizer=fallback_tokenizer,
                seed=seed + (hash(name) & 0xFFFF_FFFF),
                shuffle_buffer=shuffle_buffer,
                batch_texts=batch_texts,
                max_doc_tokens=max_doc_tokens,
                min_doc_tokens=min_doc_tokens,
                chunk_overlap=cfg.chunk_overlap,
            )
            for name in names
        }

        produced = 0

        while produced < total_batches:
            batch_tokens, batch_segids = [], []

            for _ in range(cfg.batch_size):
                tokens = [0] * cfg.chunk_size
                segids = [-1] * cfg.chunk_size

                pos = 0
                seg = 0

                while pos < cfg.chunk_size:
                    name = rng.choices(names, weights=weights, k=1)[0]
                    src = streams[name]

                    piece, ended = src.take_up_to(cfg.chunk_size - pos)
                    if not piece:
                        continue

                    n = len(piece)
                    tokens[pos:pos+n] = piece
                    segids[pos:pos+n] = (seg,) * n
                    pos += n

                    if pos >= cfg.chunk_size and not ended:
                        src.rewind_overlap_if_mid_doc()

                    seg += 1

                batch_tokens.append(tokens)
                batch_segids.append(segids)

            input_ids = torch.tensor(batch_tokens, dtype=torch.long)
            seg_ids = torch.tensor(batch_segids, dtype=torch.int32)
            # block_mask = _build_block_diag_causal_mask(seg_ids, causal)

            produced += 1
            yield {"input_ids": input_ids, "seg_ids": seg_ids}

    ds = IterableDataset.from_generator(generator)

    # ---- materialize + upload if requested ----
    if cfg.hf_dataset_name is not None:
        if cfg.hf_token is not None:
            os.environ["HF_TOKEN"] = cfg.hf_token

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = Dataset.from_generator(
                generator,
                writer_batch_size=64,        # batches per write (small = low RAM)
                cache_dir=tmpdir,
            )

            ds.save_to_disk(tmpdir, max_shard_size="2gb")
            ds = Dataset.load_from_disk(tmpdir)

            ds.push_to_hub(cfg.hf_dataset_name)
            return ds
    return ds


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

    _ = build_llm_dataset(cfg, tokenizerFast, tokenizerSlow)

if __name__ == "__main__":
    main()