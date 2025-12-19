import shutil
import torch

from config import *
from ultralytics import YOLO
from datetime import datetime


def train_model(resume=False):
    # === Model Selection ===
    base_model = NT_MODEL_PATH if not resume else T_MODEL_PATH
    model = YOLO(str(base_model))

    # === Dataset Config ===
    data_config = DATA_DIR / "data.yaml"

    # === Unique run name with datetime ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"yolo11s_seg_{timestamp}"

    # === Set epochs depending on whether it's incremental retraining ===
    num_epochs = 100 if not resume else 15  # 100 for full, 15 for active learning

    # === Training ===
    model.train(
        data=str(data_config),
        epochs=num_epochs,
        imgsz=512,
        project=str(RUNS_DIR),
        name=run_name,
        batch=8,
        workers=4,
        task="segment",
        resume=resume,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # === Save final model safely ===
    best_model_path = RUNS_DIR / run_name / "weights" / "best.pt"

    trained_model_path = MODEL_DIR / "yolo11s-seg-trained.pt"

    if best_model_path.exists():
        shutil.copy(best_model_path, trained_model_path)
        print(f"\n✅ Training complete. Model copied to: {trained_model_path}")
    else:
        raise FileNotFoundError("❌ best.pt not found after training")


if __name__ == "__main__":
    train_model(resume=False)
