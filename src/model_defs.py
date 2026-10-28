import sys
from pathlib import Path

MODEL_ZOO_CODE = Path(__file__).resolve().parents[2] / "ModelZooDataset" / "code"
sys.path.append(str(MODEL_ZOO_CODE))

from model_definitions.def_net import CNN, NNmodule

__all__ = ["CNN", "NNmodule", "build_model_from_checkpoint", "evaluate"]


def build_model_from_checkpoint(config: dict, state_dict: dict, seed: int = 42) -> NNmodule:
    """Build a Model Zoos model and load its checkpoint parameters."""
    model = NNmodule(config, cuda=False, seed=seed, verbosity=0)
    model.model.load_state_dict(state_dict)
    model.model.eval()
    return model


def evaluate(model: NNmodule, dataloader):
    """Return loss and accuracy for a dataloader."""
    return model.test_epoch(dataloader, epoch=0)
