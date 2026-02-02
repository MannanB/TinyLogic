from pydantic import BaseModel

from typing import Optional

class TinyLogicLMConfig(BaseModel):
    training_mode: str = "pretrain"  # "pretrain" or "sft"
    # Model
    model_name: str = "tinylogic-base"
    hidden_size: int = 256
    intermediate_size: int = 64
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    dropout_rate: float = 0.1

    chunk_size: int = 256
    chunk_overlap: int = 0
    total_tokens: int = 100_000_000
    batch_size: int = 2
    grad_accum_steps: int = 16
    percent_tinystories: float = 0.4
    percent_math: float = 0.267
    percent_gsm8k: float = 0.133
    percent_dclm: float = 0.1
    percent_fineweb: float = 0.1

    lr_max: float = 8e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    adam_weight_decay: float = 1e-1
    warmup_ratio: float = 0.02

    hf_token: Optional[str] = None  # HuggingFace token if needed
    hf_dataset_name: Optional[str] = None

    test_samples_dataset: Optional[int] = None
