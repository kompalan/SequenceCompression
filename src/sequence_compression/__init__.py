from .module import SIGReg, TransformerDecoder, TransformerEncoder, Transpressor, JEPA


def train() -> None:
    from .train import train

    train()

def eval() -> None:
    from .eval.eval import eval

    eval()

def train_observation_decoder() -> None:
    from .eval.observation_decoder import train_probe, parse_args
    train_probe(parse_args())
    
__all__ = ["SIGReg", "TransformerDecoder", "TransformerEncoder", "Transpressor", "JEPA"]