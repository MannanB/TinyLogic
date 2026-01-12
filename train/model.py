
from transformers import Gemma3Config, Gemma3ForCausalLM, Gemma3TextConfig
from .config import TinyLogicLMConfig


def load_model(
    tinyLMConfig: TinyLogicLMConfig,
    vocab_size: int = 32_000,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    bos_token_id: int = 2,
):
    text_config = Gemma3TextConfig(  # note tied word embeddings
        vocab_size=vocab_size,
        hidden_size=tinyLMConfig.hidden_size,
        intermediate_size=tinyLMConfig.intermediate_size,
        num_hidden_layers=tinyLMConfig.num_hidden_layers,
        num_attention_heads=tinyLMConfig.num_attention_heads,
        num_key_value_heads=tinyLMConfig.num_key_value_heads,
        head_dim=tinyLMConfig.head_dim,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        hidden_dropout_prob=tinyLMConfig.dropout_rate,
        attention_dropout=tinyLMConfig.dropout_rate,
        attn_implementation="sdpa" # using torch.nn.attention sdpa_kernel to use flash attention (hopefully this works lol)
    )
    # config = Gemma3Config(text_config=text_config)
    model = Gemma3ForCausalLM(text_config)

    print(f"Model loaded with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    return model
