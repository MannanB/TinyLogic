
from .config import TinyLogicLMConfig
from .train import train
import os

def main(run, config):
    os.system("pip install zstandard")
    cfg = TinyLogicLMConfig(**config)

    train(run, cfg)

