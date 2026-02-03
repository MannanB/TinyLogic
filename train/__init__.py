
from .config import TinyLogicLMConfig
import os

def main(run, config):
    cfg = TinyLogicLMConfig(**config)
    if cfg.training_mode == "pretrain":
        from .train import train
        train(run, cfg)
    elif cfg.training_mode == "sft":
        from .SFT import train
        train(run, cfg)

