from transformers import DebertaV2Tokenizer

hf_tokenizer = DebertaV2Tokenizer(
    vocab_file="./tlm_bpe_8k.model",
    max_len=16384
)

save_path = './hf_tokenizer'
hf_tokenizer.save_pretrained(save_path)