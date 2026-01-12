
from .config import MicroLMConfig
from .train import train

def main(run, config):

    cfg = MicroLMConfig(**config)

    train(run, cfg)