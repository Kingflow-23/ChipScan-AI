import torch

from segment_anything.segment_anything.build_sam import sam_model_registry
from segment_anything.segment_anything.predictor import SamPredictor
from config import *


def load_sam_model(
    model_type: str = "vit_h", checkpoint_path: str = None, device: str = None
):
    """
    Loads the SAM model from a checkpoint.

    Args:
        model_type (str): Type of SAM model to load (e.g., 'vit_h', 'vit_b', 'vit_l').
        checkpoint_path (str, optional): Path to the SAM checkpoint file.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to GPU if available.

    Returns:
        model: The loaded SAM model.
    """
    try:
        if checkpoint_path is None:
            checkpoint_path = SAM_CHECKPOINT_PATH

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # Load the SAM model
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device)
        sam.eval()

        print(
            f"[INFO] SAM model '{model_type}' loaded successfully on {device} from '{checkpoint_path}'"
        )
        return sam

    except Exception as e:
        raise RuntimeError(f"Failed to load SAM model: {e}")


def initialize_predictor(model, device: str = None):
    """
    Initializes the SAM predictor with the given model.

    Args:
        model: A loaded SAM model.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to GPU if available.

    Returns:
        predictor: An instance of SamPredictor.
    """
    try:
        predictor = SamPredictor(model)

        # Determine device
        if device is None:
            device = next(model.parameters()).device
        else:
            device = torch.device(device)

        predictor.device = device

        print(f"[INFO] SAM Predictor initialized on {device}")
        return predictor

    except Exception as e:
        raise RuntimeError(f"Failed to initialize SAM predictor: {e}")
