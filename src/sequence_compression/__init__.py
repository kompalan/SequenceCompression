from .module import SIGReg, TransformerDecoder, TransformerEncoder, Transpressor, JEPA

def main() -> None:
    from .train import train

    train()

__all__ = ["SIGReg", "TransformerDecoder", "TransformerEncoder", "Transpressor", "JEPA"]