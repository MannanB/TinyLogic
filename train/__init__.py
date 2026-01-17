
from .config import TinyLogicLMConfig
from .train import train

def main(run, config):

    cfg = TinyLogicLMConfig(**config)

    train(run, cfg)

