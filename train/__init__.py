
from .config import TinyLogicLMConfig
from .train import train
import os

def main(run, config):
    cfg = TinyLogicLMConfig(**config)

    train(run, cfg)

