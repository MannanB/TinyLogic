
import argparse
import json
import torch
from transformers import AutoTokenizer

from .config import TinyLogicLMConfig
from .model import load_model
from pathlib import Path


# python -m container.projects.microlm.test --config-path ./inputs/microlm-50m.json --weights-path ./microlm.pt --prompt "test prompt haha" --max-new-tokens 256

def load_cfg(path: str) -> TinyLogicLMConfig:
    with open(path, "r") as f:
        data = json.load(f)
    return TinyLogicLMConfig(**data.get("input", data).get("config", data))


def prepare_tokenizer():
    file_path = Path(__file__).resolve()
    tokenizer_path = file_path.parent / "tokenizer" / "hf_tokenizer"
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    tok.padding_side = "right"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.config_path)
    tok = prepare_tokenizer()

    model = load_model(
        cfg,
        vocab_size=len(tok),
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id or tok.pad_token_id,
        bos_token_id=tok.bos_token_id or tok.eos_token_id or tok.pad_token_id,
    )
    state = torch.load(args.weights_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attn = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    print(tok.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
