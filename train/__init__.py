
from .config import TinyLogicLMConfig
from .train import train
import os

def main(run, config):
    cfg = TinyLogicLMConfig(**config)
    if cfg.training_mode == "pretrain":

        train(run, cfg)
    elif cfg.training_mode == "sft":
        from .SFT import sft_train
        sft_train(run, './train/out_microlm.pt', cfg)

