import os
import sentencepiece as spm
from tqdm import tqdm
from dataset import text_stream

VOCAB_SIZE = 8192
MODEL_PREFIX = "tlm_bpe_8k"
CORPUS_PATH = "corpus.txt"

TOTAL_CHARS = 37_500_000

DATASETS = {
    "math": 0.4 * 2 / 3,
    "gsm8k": 0.4 / 3,
    "tinystories": 0.3,
    "fineweb": 0.3,
}

def build_corpus():
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for name, frac in DATASETS.items():
            char_budget = int(TOTAL_CHARS * frac)
            chars_written = 0

            pbar = tqdm(
                total=char_budget,
                unit="char",
                unit_scale=True,
                desc=f"Building corpus: {name}",
            )
            total_samples = 0
            for text in text_stream(name, split="train", streaming=True):
                if not text or not text.strip():
                    continue

                text = text.replace("\n", " ")
                f.write(text + "\n")

                n = len(text)
                chars_written += n
                pbar.update(n)
                total_samples += 1

                if chars_written >= char_budget:
                    break

            pbar.close()
            print(f"Achieved {total_samples} samples for {name}")

def train_spm():
    spm.SentencePieceTrainer.train(
        input=CORPUS_PATH,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        byte_fallback=True,
        character_coverage=1.0,
        normalization_rule_name="identity",
        split_digits=True,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
    )

if not os.path.exists(CORPUS_PATH):
    build_corpus()
else:
    print("Corpus already exists, skipping build.")

if not os.path.exists(f"{MODEL_PREFIX}.model"):
    train_spm()
else:
    print("Tokenizer already trained, skipping.")
