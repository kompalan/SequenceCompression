from .module import SIGReg, TransformerDecoder, TransformerEncoder, Transpressor, JEPA

def train() -> None:
    from .train import train

    train()

def eval() -> None:
    from .eval.eval import eval

    eval()
    
__all__ = ["SIGReg", "TransformerDecoder", "TransformerEncoder", "Transpressor", "JEPA"]