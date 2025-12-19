import cv2
import matplotlib.pyplot as plt

from config import *
from ultralytics import YOLO

model = YOLO(str(T_MODEL_PATH))


def run_inference(image_path: str, conf: float = 0.25):
    """
    Runs YOLO11s segmentation inference on a single image.
    Returns Ultralytics results object.
    """
    results = model.predict(source=image_path, task="segment", conf=conf, save=False)
    return results


def show_results(results):
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.imshow(annotated)
    plt.axis("off")
    plt.show()


# print(run_inference(r"C:\Users\flori\OneDrive - aivancity\4th year\Deployment & Maintenance of AI Models\Group Work\pack\pack\app\dataset\valid\images\06_jpg.rf.622bf1347fb91bb265c341fe98cf35a6.jpg"))

# show_results(run_inference(r"C:\Users\flori\OneDrive - aivancity\4th year\Deployment & Maintenance of AI Models\Group Work\pack\pack\app\dataset\valid\images\06_jpg.rf.622bf1347fb91bb265c341fe98cf35a6.jpg"))
