import shutil
import torch

from ultralytics import YOLO
from datetime import datetime

from config import *


def train_model(retrain=False):
    # === Model Selection ===
    base_model = NT_MODEL_PATH if not retrain else T_MODEL_PATH
    model = YOLO(str(base_model))

    # === Dataset Config ===
    data_config = DATA_DIR / "data.yaml"

    # === Unique run name with datetime ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{MODEL_NAME}_{timestamp}"

    # === Set epochs depending on whether it's incremental retraining ===
    num_epochs = 100 if not retrain else 50  # 100 for full, 50 for active learning

    if retrain:
        for label_file in LABELS_DIR.glob("*.txt"):
            image_id = label_file.stem
            # Find the corresponding image
            possible_images = list(UPLOAD_DIR.glob(f"{image_id}.*"))
            if not possible_images:
                continue
            image_file = possible_images[0]

            # Copy to dataset train folders
            shutil.copy(image_file, TRAIN_IMAGE_DATA_DIR / image_file.name)
            shutil.copy(label_file, TRAIN_LABELS_DATA_DIR / label_file.name)

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
        resume=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # === Save final model safely ===
    best_model_path = RUNS_DIR / run_name / "weights" / "best.pt"

    if best_model_path.exists():
        shutil.copy(best_model_path, T_MODEL_PATH)
        print(f"\n✅ Training complete. Model copied to: {T_MODEL_PATH}")
    else:
        raise FileNotFoundError("❌ best.pt not found after training")


if __name__ == "__main__":
    train_model(retrain=False)
