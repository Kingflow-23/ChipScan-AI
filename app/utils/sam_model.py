import torch

from segment_anything.segment_anything.build_sam import sam_model_registry
from segment_anything.segment_anything.predictor import SamPredictor
from config import SAM_CHECKPOINT_PATH


def load_sam_model(model_type: str = "vit_h"):
    """
    Loads the SAM model and moves it to the appropriate device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    sam.eval()

    print(f"[INFO] SAM model loaded on {device}")
    return sam


def initialize_predictor(sam_model):
    """
    Initializes a SAM predictor from a loaded SAM model.
    """
    predictor = SamPredictor(sam_model)
    print("[INFO] SAM Predictor initialized")
    return predictor
