import argparse
import json
import torch
from transformers import AutoTokenizer
from pathlib import Path

from .config import TinyLogicLMConfig
from .model import load_model
from .SFT import patch_vocab   # <-- IMPORTANT


# python -m container.projects.microlm.test \
#   --config-path ./inputs/microlm-50m.json \
#   --weights-path ./microlm.pt


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


def format_chat(conversation):
    """
    conversation: list of {"role": str, "content": str}
    """
    out = []
    for msg in conversation:
        out.append(
            f"<|im_start|>{msg['role']}\n"
            f"{msg['content']}"
            f"<|im_end|>\n"
        )
    return "".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # Load config + tokenizer
    # ------------------------------------------------------------
    cfg = load_cfg(args.config_path)
    tok = prepare_tokenizer()

    # ------------------------------------------------------------
    # Build model with *current* tokenizer size
    # ------------------------------------------------------------
    model = load_model(
        cfg,
        vocab_size=len(tok),
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id or tok.pad_token_id,
        bos_token_id=tok.bos_token_id or tok.eos_token_id or tok.pad_token_id,
    )

    # ------------------------------------------------------------
    # Patch vocab for chat tokens (<|im_start|>, <|im_end|>)
    # ------------------------------------------------------------
    patch_vocab(model, tok)

    # ------------------------------------------------------------
    # Load weights AFTER patching vocab
    # ------------------------------------------------------------
    state = torch.load(args.weights_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    # ------------------------------------------------------------
    # Interactive chat loop
    # ------------------------------------------------------------
    conversation = []

    print("\n🧠 TinyLogic Chat — type 'exit' to quit\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        conversation.append({"role": "user", "content": user_input})

        prompt = format_chat(conversation) + "<|im_start|>assistant\n"

        inputs = tok(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        string_tokenized = tok.decode(inputs["input_ids"][0], skip_special_tokens=False)
        token_str_to_id_map = {tok.convert_ids_to_tokens([id])[0]: id for id in inputs["input_ids"][0]}
        print(string_tokenized)
        print(token_str_to_id_map)
        # print(f"String tokenized: {string_tokenized}")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        # Only decode newly generated tokens
        new_tokens = outputs[0][input_ids.shape[1]:]
        assistant_text = tok.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"Assistant: {assistant_text}\n")

        conversation.append(
            {"role": "assistant", "content": assistant_text}
        )


if __name__ == "__main__":
    main()
