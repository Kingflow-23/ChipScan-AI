import threading

from utils.training_yolo import train_model


def start_retraining(resume: bool = True):
    """
    Starts YOLO retraining in a background thread.
    """
    thread = threading.Thread(
        target=train_model,
        kwargs={"resume": resume},
        daemon=True,
    )
    thread.start()
