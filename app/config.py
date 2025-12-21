from pathlib import Path

CHIP_CLASS = 0
VOID_CLASS = 1

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model"
DATA_DIR = BASE_DIR.parent / "app" / "dataset"

TRAIN_IMAGE_DATA_DIR = DATA_DIR / "train" / "images"
TRAIN_LABELS_DATA_DIR = DATA_DIR / "train" / "labels"

RUNS_DIR = BASE_DIR.parent / "runs"

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
RESULT_JSON_DIR = BASE_DIR / "static" / "results_json"
LABELS_DIR = BASE_DIR / "static" / "labels"

NT_MODEL_PATH = MODEL_DIR / "yolo11s-seg.pt"
T_MODEL_PATH = MODEL_DIR / "yolo11s-seg-trained.pt"
SAM_CHECKPOINT_PATH = MODEL_DIR / "sam_vit_b_01ec64.pth"
